import cv2
import numpy as np

from .geometry import make_T

DH = [(0.0, 0.333, 0.0), (0.0, 0.0, -np.pi / 2), (0.0, 0.316, np.pi / 2), (0.0825, 0.0, np.pi / 2),
      (-0.0825, 0.384, -np.pi / 2), (0.0, 0.0, np.pi / 2), (0.088, 0.0, np.pi / 2)]
FLANGE = 0.107


def _dh(a, d, alpha, theta):
    """Modified (Craig) DH link transform."""
    ca, sa, ct, st = np.cos(alpha), np.sin(alpha), np.cos(theta), np.sin(theta)
    return np.array([[ct, -st, 0.0, a], [st * ca, ct * ca, -sa, -d * sa], [st * sa, ct * sa, ca, d * ca],
                     [0.0, 0.0, 0.0, 1.0]])


def _ee_offset():
    """Flange to the EE frame the arm state reports (hand frame, -45 deg about z)."""
    c, s = np.cos(-np.pi / 4), np.sin(-np.pi / 4)
    return make_T(np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]]), [0.0, 0.0, FLANGE])


EE = _ee_offset()


def fk_frames(q):
    """Joint frames (z axis = joint axis) and the EE pose in the robot base frame."""
    M, frames = np.eye(4), []
    for (a, d, al), th in zip(DH, q):
        M = M @ _dh(a, d, al, th)
        frames.append(M)
    return frames, M @ EE


def jacobian(q):
    """Geometric 6x7 Jacobian of the EE in the base frame and the EE pose."""
    frames, T = fk_frames(q)
    p = T[:3, 3]
    J = np.zeros((6, 7))
    for i, F in enumerate(frames):
        z = F[:3, 2]
        J[:3, i] = np.cross(z, p - F[:3, 3])
        J[3:, i] = z
    return J, T


def _err(T, T_goal):
    """6D pose error (position, rotation vector) from T to T_goal."""
    r, _ = cv2.Rodrigues(T_goal[:3, :3] @ T[:3, :3].T)
    return np.r_[T_goal[:3, 3] - T[:3, 3], r.ravel()]


def _interp(T0, T1, s):
    """Pose a fraction s along the straight line (and shortest rotation) from T0 to T1."""
    r, _ = cv2.Rodrigues(T0[:3, :3].T @ T1[:3, :3])
    R, _ = cv2.Rodrigues(r * s)
    return make_T(T0[:3, :3] @ R, T0[:3, 3] + s * (T1[:3, 3] - T0[:3, 3]))


class Arm:
    """FR3 kinematics in the world frame with joint limits, to check a move before commanding it."""

    def __init__(self, spec):
        self.T_base = make_T(np.eye(3), spec.get("base_pos", [0.0, 0.0, 0.0]))
        self.T_base_inv = np.linalg.inv(self.T_base)
        self.q_min = np.asarray(spec["q_min"], dtype=float)
        self.q_max = np.asarray(spec["q_max"], dtype=float)
        self.margin = float(spec.get("margin_rad", 0.25))

    def fk(self, q):
        """EE pose in the world frame."""
        return self.T_base @ fk_frames(q)[1]

    def follow(self, q, T_goal, steps=20, iters=4, tol=(1e-3, 1e-2)):
        """Joints after the straight Cartesian move to T_goal, tracked by resolved rate like the impedance controller.

        Returns None if a joint gets closer to a limit than the margin (a joint already inside the margin may
        not go further) or the goal is not reached."""
        q = np.asarray(q, dtype=float).copy()
        lo = np.minimum(self.q_min + self.margin, q)
        hi = np.maximum(self.q_max - self.margin, q)
        T_start = self.fk(q)
        for k in range(1, steps + 1):
            T_k = self.T_base_inv @ _interp(T_start, T_goal, k / steps)
            for _ in range(iters):
                J, T = jacobian(q)
                q = q + J.T @ np.linalg.solve(J @ J.T + 1e-4 * np.eye(6), _err(T, T_k))
            if np.any(q < lo) or np.any(q > hi):
                return None
        e = _err(fk_frames(q)[1], self.T_base_inv @ T_goal)
        return q if np.linalg.norm(e[:3]) < tol[0] and np.linalg.norm(e[3:]) < tol[1] else None
