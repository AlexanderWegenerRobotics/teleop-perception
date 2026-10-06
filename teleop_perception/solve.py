import cv2
import numpy as np
from scipy.optimize import least_squares

from .geometry import T_from_rvec, average_poses, inv_T, make_T, rvec_from_T

DIST_TERMS = (0, 2, 4, 5)


def usable(points, img):
    """True if a view has enough non-collinear target points for a pose."""
    m = ~np.isnan(img[:, 0])
    if m.sum() < 4:
        return False
    P = points[m, :2] - points[m, :2].mean(axis=0)
    s = np.linalg.svd(P, compute_uv=False)
    return s[1] > 0.1 * s[0]


def view_pose(points, img, K, dist):
    """Camera-from-target pose of one view from its visible points (IPPE, lower error)."""
    m = ~np.isnan(img[:, 0])
    n, rvecs, tvecs, errs = cv2.solvePnPGeneric(points[m], img[m], K, dist, flags=cv2.SOLVEPNP_IPPE)
    k = int(np.argmin(np.ravel(errs)[:n]))
    return T_from_rvec(rvecs[k], tvecs[k])


def hand_eye_init(T_we, T_ct):
    """Closed-form eye-to-hand start (Park & Martin on A Y = Y B): returns (T_world_cam, T_ee_target)."""
    M = np.zeros((3, 3))
    rows, rhs, pairs = [], [], []
    n = len(T_we)
    for i in range(n):
        for j in range(i + 1, n):
            A = inv_T(T_we[j]) @ T_we[i]
            B = inv_T(T_ct[j]) @ T_ct[i]
            a, _ = cv2.Rodrigues(A[:3, :3])
            b, _ = cv2.Rodrigues(B[:3, :3])
            if np.linalg.norm(a) < np.radians(2.0):
                continue
            M += b @ a.T
            pairs.append((A, B))
    if len(pairs) < 3:
        raise RuntimeError("calibration poses have too little rotation to solve the hand-eye problem")
    w, V = np.linalg.eigh(M.T @ M)
    R_y = V @ np.diag(1.0 / np.sqrt(np.maximum(w, 1e-12))) @ V.T @ M.T
    U, _, Vt = np.linalg.svd(R_y)
    R_y = U @ np.diag([1.0, 1.0, np.linalg.det(U @ Vt)]) @ Vt
    for A, B in pairs:
        rows.append(A[:3, :3] - np.eye(3))
        rhs.append(R_y @ B[:3, 3] - A[:3, 3])
    t_y = np.linalg.lstsq(np.vstack(rows), np.concatenate(rhs), rcond=None)[0]
    T_et = make_T(R_y, t_y)
    T_wc = average_poses([T @ T_et @ inv_T(C) for T, C in zip(T_we, T_ct)])
    return T_wc, T_et


def _dist_full(d, n):
    """Five OpenCV distortion terms from the first n estimated ones."""
    out = np.zeros(5)
    out[:n] = d[:n]
    return out


def solve(samples, points, K, dist, estimate_intrinsics=False, dist_terms=5):
    """Camera pose, target-on-gripper pose and optionally K/dist from [(T_world_ee, image points)] on reprojection error."""
    if dist_terms not in DIST_TERMS:
        raise ValueError(f"dist_terms must be one of {DIST_TERMS}")
    samples = [(T, img) for T, img in samples if usable(points, img)]
    dist = np.zeros(5) if dist is None else np.ravel(dist)[:5]
    T_we = [T for T, _ in samples]
    T_ct = [view_pose(points, img, K, dist) for _, img in samples]
    with np.errstate(divide="ignore", over="ignore", invalid="ignore"):
        T_wc0, T_et0 = hand_eye_init(T_we, T_ct)
    masks = [~np.isnan(img[:, 0]) for _, img in samples]
    P_h = np.c_[points, np.ones(len(points))]
    obs = np.vstack([img[m] for (_, img), m in zip(samples, masks)])
    counts = [int(m.sum()) for m in masks]
    n_d = dist_terms if estimate_intrinsics else 0

    def unpack(x):
        """Parameter vector to (T_cam_world, T_ee_target, K, dist)."""
        T_cw, T_et = T_from_rvec(x[0:3], x[3:6]), T_from_rvec(x[6:9], x[9:12])
        if not estimate_intrinsics:
            return T_cw, T_et, K, dist
        fx, fy, cx, cy = x[12:16]
        Kx = np.array([[fx, 0.0, cx], [0.0, fy, cy], [0.0, 0.0, 1.0]])
        return T_cw, T_et, Kx, _dist_full(x[16:16 + n_d], n_d)

    def project(x):
        """All visible target points projected through the current parameters."""
        T_cw, T_et, Kx, dx = unpack(x)
        P = np.vstack([((T_cw @ T @ T_et) @ P_h[m].T).T[:, :3] for T, m in zip(T_we, masks)])
        proj, _ = cv2.projectPoints(P, np.zeros(3), np.zeros(3), Kx, dx)
        return proj.reshape(-1, 2)

    def residuals(x):
        """Reprojection error over every visible point of every view."""
        return (project(x) - obs).ravel()

    r1, t1 = rvec_from_T(inv_T(T_wc0))
    r2, t2 = rvec_from_T(T_et0)
    x0 = np.r_[r1.ravel(), t1.ravel(), r2.ravel(), t2.ravel()]
    if estimate_intrinsics:
        x0 = np.r_[x0, K[0, 0], K[1, 1], K[0, 2], K[1, 2], dist[:n_d]]
    with np.errstate(divide="ignore", over="ignore", invalid="ignore"):
        sol = least_squares(residuals, x0, loss="huber", f_scale=1.0, x_scale="jac")
        J = sol.jac
        s2 = float(np.sum(sol.fun ** 2)) / max(1, J.shape[0] - J.shape[1])
        sigma = np.sqrt(np.clip(np.diag(np.linalg.pinv(J.T @ J)) * s2, 0.0, None))
    T_cw, T_et, Kx, dx = unpack(sol.x)
    e = np.linalg.norm(sol.fun.reshape(-1, 2), axis=1)
    per = np.array([np.sqrt(np.mean(c ** 2)) for c in np.split(e, np.cumsum(counts)[:-1])])
    return {"T_world_cam": inv_T(T_cw), "T_ee_target": T_et, "K": Kx, "dist": dx, "rms_px": per,
            "sigma_K": sigma[12:16] if estimate_intrinsics else None, "samples": samples}


def model_shift_px(K0, d0, K1, d1, size, n=24):
    """Pixel distance between two camera models over an image grid: (max, mean)."""
    w, h = size
    u, v = np.meshgrid(np.linspace(0, w - 1, n), np.linspace(0, h - 1, max(2, n * h // w)))
    px = np.c_[u.ravel(), v.ravel()].astype(np.float64)
    rays = cv2.undistortPoints(px.reshape(-1, 1, 2), K1, d1).reshape(-1, 2)
    P = np.c_[rays, np.ones(len(rays))]
    proj, _ = cv2.projectPoints(P, np.zeros(3), np.zeros(3), K0, d0)
    e = np.linalg.norm(proj.reshape(-1, 2) - px, axis=1)
    return float(e.max()), float(e.mean())


def cell_hits(samples, size, grid):
    """Observed target points per image cell, array (rows, cols)."""
    w, h = size
    gx, gy = grid
    hits = np.zeros((gy, gx), dtype=int)
    for _, img in samples:
        p = img[~np.isnan(img[:, 0])]
        i = np.clip((p[:, 0] / w * gx).astype(int), 0, gx - 1)
        j = np.clip((p[:, 1] / h * gy).astype(int), 0, gy - 1)
        np.add.at(hits, (j, i), 1)
    return hits


def coverage_map(hits, need):
    """Text map of the image cells: '#' covered, '+' some points, '.' none."""
    return ["".join("#" if v >= need else "+" if v > 0 else "." for v in row) for row in hits]
