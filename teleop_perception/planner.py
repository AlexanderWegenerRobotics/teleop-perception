import cv2
import numpy as np

from .geometry import inv_T, make_T, rot_angle_deg

DEPTH_STEPS = (1.0, 0.9, 1.1, 0.8, 1.2)
CENTER_PULL = (1.0, 0.8, 0.6)


def _frame_with_z(z):
    """Some rotation whose z axis is z."""
    a = np.array([1.0, 0.0, 0.0]) if abs(z[0]) < 0.9 else np.array([0.0, 1.0, 0.0])
    x = np.cross(a, z)
    x /= np.linalg.norm(x)
    return np.column_stack([x, np.cross(z, x), z])


def _closest_roll(R_ct, T_wc, T_et, R0):
    """Rotation of the target about its normal that keeps the hand closest to R0; returns (T_world_ee rot, turn deg)."""
    best, best_a = None, np.inf
    for a in np.radians(np.arange(0.0, 360.0, 5.0)):
        Rz, _ = cv2.Rodrigues(np.array([0.0, 0.0, a]))
        R_we = T_wc[:3, :3] @ R_ct @ Rz @ T_et[:3, :3].T
        d = rot_angle_deg(R_we, R0)
        if d < best_a:
            best, best_a = R_ct @ Rz, d
    return best, best_a


def _visible(points, T_ct, K, size):
    """Fraction of target points in front of the camera and inside the image."""
    P = (T_ct[:3, :3] @ points.T).T + T_ct[:3, 3]
    if np.any(P[:, 2] <= 0.05):
        return 0.0
    uv = (K @ (P / P[:, 2:3]).T).T[:, :2]
    w, h = size
    return float(np.mean((uv[:, 0] >= 0) & (uv[:, 0] < w) & (uv[:, 1] >= 0) & (uv[:, 1] < h)))


def candidates(T_wc, K, size, points, T_et, R0, cov, max_options=6):
    """Hand poses per distance and image cell that put the target there, tilted, within the box:
    [(distance index, (col, row), [T_world_ee options, least hand turn first])]."""
    gx, gy = (int(v) for v in cov.get("grid", [5, 3]))
    w, h = size
    tilt = np.radians(float(cov.get("tilt_deg", 25.0)))
    max_turn = float(cov.get("max_turn_deg", 50.0))
    min_vis = float(cov.get("min_visible", 0.6))
    min_z = float(cov.get("min_target_z", -np.inf))
    lo = np.asarray(cov.get("workspace_min", [-np.inf] * 3), dtype=float)
    hi = np.asarray(cov.get("workspace_max", [np.inf] * 3), dtype=float)
    Kinv = np.linalg.inv(K)
    P_h = np.c_[points, np.ones(len(points))]
    out = []
    for n, d in enumerate(cov.get("distances_m", [0.5])):
        for j in range(gy):
            for i in range(gx):
                u0, v0 = (i + 0.5) / gx * w, (j + 0.5) / gy * h
                opts = []
                for pull in CENTER_PULL:
                    u, v = w / 2 + (u0 - w / 2) * pull, h / 2 + (v0 - h / 2) * pull
                    ray = Kinv @ np.array([u, v, 1.0])
                    ray /= np.linalg.norm(ray)
                    for a in np.radians([45.0, 135.0, 225.0, 315.0]):
                        R_tilt, _ = cv2.Rodrigues(np.array([np.cos(a), np.sin(a), 0.0]) * tilt)
                        R_ct, turn = _closest_roll(_frame_with_z(R_tilt @ -ray), T_wc, T_et, R0)
                        if turn > max_turn:
                            continue
                        for s in DEPTH_STEPS:
                            T_ct = make_T(R_ct, ray * float(d) * s)
                            T_we = T_wc @ T_ct @ inv_T(T_et)
                            p = T_we[:3, 3]
                            if np.any(p < lo) or np.any(p > hi):
                                continue
                            if (T_wc @ T_ct @ P_h.T)[2].min() < min_z or _visible(points, T_ct, K, size) < min_vis:
                                continue
                            opts.append((pull < 1.0, turn, T_we))
                            break
                if opts:
                    opts.sort(key=lambda o: (o[0], o[1]))
                    out.append((n, (i, j), [o[2] for o in opts[:max_options]]))
    return out


def motion_cost(T_a, T_b, m_per_rad=0.2):
    """Travel between two hand poses: metres plus rotation weighted as metres per radian."""
    return float(np.linalg.norm(T_a[:3, 3] - T_b[:3, 3]) + m_per_rad * np.radians(rot_angle_deg(T_a[:3, :3], T_b[:3, :3])))


def order_poses(poses, T_start):
    """Nearest-neighbour order on position and rotation, to keep travel short."""
    poses, out, cur = list(poses), [], T_start
    while poses:
        k = int(np.argmin([motion_cost(cur, P) for P in poses]))
        cur = poses.pop(k)
        out.append(cur)
    return out
