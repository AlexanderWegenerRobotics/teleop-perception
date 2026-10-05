import numpy as np
import cv2

AXES = {
    "+x": (1.0, 0.0, 0.0), "-x": (-1.0, 0.0, 0.0),
    "+y": (0.0, 1.0, 0.0), "-y": (0.0, -1.0, 0.0),
    "+z": (0.0, 0.0, 1.0), "-z": (0.0, 0.0, -1.0),
}


def make_T(R, t):
    """Builds a 4x4 transform from rotation and translation."""
    T = np.eye(4)
    T[:3, :3] = R
    T[:3, 3] = np.asarray(t, dtype=float).reshape(3)
    return T


def inv_T(T):
    """Inverts a rigid 4x4 transform."""
    R = T[:3, :3]
    return make_T(R.T, -R.T @ T[:3, 3])


def T_from_rvec(rvec, tvec):
    """Converts an OpenCV rvec/tvec pair to a 4x4 transform."""
    R, _ = cv2.Rodrigues(np.asarray(rvec, dtype=float).reshape(3, 1))
    return make_T(R, np.asarray(tvec, dtype=float).reshape(3))


def rvec_from_T(T):
    """Converts a 4x4 transform to an OpenCV rvec/tvec pair."""
    rvec, _ = cv2.Rodrigues(T[:3, :3])
    return rvec.reshape(3, 1), T[:3, 3].reshape(3, 1).copy()


def quat_from_R(R):
    """Rotation matrix to unit quaternion (w, x, y, z) with w >= 0."""
    m = R
    tr = m[0, 0] + m[1, 1] + m[2, 2]
    if tr > 0.0:
        s = 2.0 * np.sqrt(tr + 1.0)
        q = np.array([0.25 * s, (m[2, 1] - m[1, 2]) / s, (m[0, 2] - m[2, 0]) / s, (m[1, 0] - m[0, 1]) / s])
    elif m[0, 0] > m[1, 1] and m[0, 0] > m[2, 2]:
        s = 2.0 * np.sqrt(1.0 + m[0, 0] - m[1, 1] - m[2, 2])
        q = np.array([(m[2, 1] - m[1, 2]) / s, 0.25 * s, (m[0, 1] + m[1, 0]) / s, (m[0, 2] + m[2, 0]) / s])
    elif m[1, 1] > m[2, 2]:
        s = 2.0 * np.sqrt(1.0 + m[1, 1] - m[0, 0] - m[2, 2])
        q = np.array([(m[0, 2] - m[2, 0]) / s, (m[0, 1] + m[1, 0]) / s, 0.25 * s, (m[1, 2] + m[2, 1]) / s])
    else:
        s = 2.0 * np.sqrt(1.0 + m[2, 2] - m[0, 0] - m[1, 1])
        q = np.array([(m[1, 0] - m[0, 1]) / s, (m[0, 2] + m[2, 0]) / s, (m[1, 2] + m[2, 1]) / s, 0.25 * s])
    q /= np.linalg.norm(q)
    return q if q[0] >= 0.0 else -q


def R_from_quat(q):
    """Unit quaternion (w, x, y, z) to rotation matrix."""
    w, x, y, z = np.asarray(q, dtype=float) / np.linalg.norm(q)
    return np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
        [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
        [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
    ])


def T_from_pos_quat(pos, quat):
    """Position plus quaternion (w, x, y, z) to a 4x4 transform."""
    return make_T(R_from_quat(quat), pos)


def R_from_rpy(rpy):
    """Roll-pitch-yaw (fixed-axis xyz, radians) to rotation matrix."""
    r, p, y = rpy
    Rx = np.array([[1, 0, 0], [0, np.cos(r), -np.sin(r)], [0, np.sin(r), np.cos(r)]])
    Ry = np.array([[np.cos(p), 0, np.sin(p)], [0, 1, 0], [-np.sin(p), 0, np.cos(p)]])
    Rz = np.array([[np.cos(y), -np.sin(y), 0], [np.sin(y), np.cos(y), 0], [0, 0, 1]])
    return Rz @ Ry @ Rx


def R_from_face_up(face, up):
    """Tag frame whose z points out of the face and whose y points along 'up'."""
    z = np.array(AXES[face])
    y = np.array(AXES[up])
    if abs(float(z @ y)) > 1e-9:
        raise ValueError(f"face {face} and up {up} must be perpendicular")
    x = np.cross(y, z)
    return np.column_stack([x, y, z])


def T_from_spec(spec):
    """Pose from a config entry with pos plus face/up, rpy or quat."""
    pos = spec.get("pos", [0.0, 0.0, 0.0])
    if "face" in spec:
        R = R_from_face_up(spec["face"], spec.get("up", "+z"))
    elif "rpy" in spec:
        R = R_from_rpy(spec["rpy"])
    elif "quat" in spec:
        R = R_from_quat(spec["quat"])
    elif "T" in spec:
        return np.asarray(spec["T"], dtype=float).reshape(4, 4)
    else:
        R = np.eye(3)
    return make_T(R, pos)


def _look_at_axes(pos, look_at):
    """Right, up and forward axes the simulator's lookAtToQuat uses."""
    f = np.asarray(look_at, dtype=float) - np.asarray(pos, dtype=float)
    f /= np.linalg.norm(f)
    w = np.array([0.0, 0.0, 1.0]) if abs(f[2]) <= 0.999 else np.array([1.0, 0.0, 0.0])
    r = np.cross(f, w)
    r /= np.linalg.norm(r)
    u = np.cross(r, f)
    return r, u, f


def T_world_cam_look_at(pos, look_at):
    """World-from-camera transform (OpenCV camera axes) for a sim look_at camera."""
    r, u, f = _look_at_axes(pos, look_at)
    return make_T(np.column_stack([r, -u, f]), pos)


def mj_quat_look_at(pos, look_at):
    """MuJoCo camera quaternion (w, x, y, z) for a look_at camera."""
    r, u, f = _look_at_axes(pos, look_at)
    return quat_from_R(np.column_stack([r, u, -f]))


def rot_angle_deg(R_a, R_b):
    """Angle of the relative rotation between two rotation matrices, degrees."""
    c = (np.trace(R_a.T @ R_b) - 1.0) / 2.0
    return float(np.degrees(np.arccos(np.clip(c, -1.0, 1.0))))


def pose_error(T_est, T_ref):
    """Translation error (m) and rotation error (deg) between two poses."""
    return float(np.linalg.norm(T_est[:3, 3] - T_ref[:3, 3])), rot_angle_deg(T_est[:3, :3], T_ref[:3, :3])


def average_poses(Ts, weights=None):
    """Weighted mean of poses: weighted mean position, eigenvector quaternion mean."""
    Ts = list(Ts)
    w = np.ones(len(Ts)) if weights is None else np.asarray(weights, dtype=float)
    w = w / w.sum()
    t = sum(wi * T[:3, 3] for wi, T in zip(w, Ts))
    M = np.zeros((4, 4))
    for wi, T in zip(w, Ts):
        q = quat_from_R(T[:3, :3])
        M += wi * np.outer(q, q)
    _, vecs = np.linalg.eigh(M)
    return make_T(R_from_quat(vecs[:, -1]), t)


def to_list(T):
    """4x4 transform as nested lists for YAML."""
    return [[float(v) for v in row] for row in np.asarray(T)]
