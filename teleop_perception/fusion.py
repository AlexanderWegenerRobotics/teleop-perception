from collections import deque
from dataclasses import dataclass

import cv2
import numpy as np
from scipy.optimize import least_squares

from .geometry import T_from_rvec, average_poses, inv_T, pose_error, rvec_from_T


@dataclass
class WorldEstimate:
    name: str
    T_world_obj: np.ndarray
    capture_ns: int
    cameras: list
    n_tags: int
    rms_px: float
    weight: float
    ambiguous: bool
    static: bool = False
    n_avg: int = 1


def weight_of(est, rms_floor):
    """Trust in one camera's estimate: more corners and lower reprojection error weigh more."""
    w = est.n_corners / max(est.rms_px, rms_floor) ** 2
    return w * (0.1 if est.ambiguous else 1.0)


def joint_refine(T0, views):
    """One object pose that minimises the reprojection error in all cameras at once.

    views: list of (Estimate, T_cam_world, K, dist).
    """
    rvec0, tvec0 = rvec_from_T(T0)

    def residuals(x):
        T = T_from_rvec(x[:3], x[3:])
        res = []
        for e, T_cw, K, dist in views:
            rv, tv = rvec_from_T(T_cw @ T)
            proj, _ = cv2.projectPoints(e.obj_pts, rv, tv, K, dist)
            res.append((proj.reshape(-1, 2) - e.img_pts).ravel())
        return np.concatenate(res)

    sol = least_squares(residuals, np.r_[rvec0.ravel(), tvec0.ravel()], loss="huber", f_scale=1.0,
                        x_scale="jac", max_nfev=50)
    r = sol.fun.reshape(-1, 2)
    return T_from_rvec(sol.x[:3], sol.x[3:]), float(np.sqrt(np.mean(np.sum(r ** 2, axis=1))))


def _view_rms(T_world_obj, view):
    """Reprojection rms of one camera's corners under a world pose."""
    e, T_cw, K, dist = view
    rv, tv = rvec_from_T(T_cw @ T_world_obj)
    proj, _ = cv2.projectPoints(e.obj_pts, rv, tv, K, dist)
    return float(np.sqrt(np.mean(np.sum((proj.reshape(-1, 2) - e.img_pts) ** 2, axis=1))))


def _joint(items, params):
    """Tries every camera's pose as seed, keeps the views that agree, returns the best joint solution."""
    gate = float(params.get("view_gate_px", 2.0))
    views = [(x[0], inv_T(x[3].T_world_cam), x[3].intrinsics.K, x[3].intrinsics.dist) for x in items]
    best = None
    for x in items:
        T, _ = joint_refine(x[1], views)
        keep = [k for k, v in enumerate(views) if _view_rms(T, v) <= gate]
        if not keep:
            continue
        if len(keep) < len(views):
            T, _ = joint_refine(T, [views[k] for k in keep])
        rms = float(np.sqrt(np.mean([_view_rms(T, views[k]) ** 2 for k in keep])))
        score = (len(keep), -rms)
        if best is None or score > best[0]:
            best = (score, T, rms, keep)
    if best is None:
        return None
    return best[1], best[2], [items[k] for k in best[3]]


def fuse(name, per_camera, params):
    """Combines one object's per-camera estimates [(Estimate, T_world_obj, capture_ns, CameraSpec)]."""
    rms_floor = float(params.get("rms_floor_px", 0.15))
    mode = params.get("mode", "joint")
    items = [(e, T, t, cam, weight_of(e, rms_floor)) for e, T, t, cam in per_camera]
    items.sort(key=lambda x: -x[4])
    used, T, rms = [items[0]], items[0][1], items[0][0].rms_px
    if len(items) > 1 and mode == "joint":
        sol = _joint(items, params)
        if sol is not None:
            T, rms, used = sol
    elif len(items) > 1:
        max_gap = float(params.get("max_cam_disagreement_m", 0.01))
        used = [x for x in items if np.linalg.norm(x[1][:3, 3] - items[0][1][:3, 3]) <= max_gap]
        T = average_poses([x[1] for x in used], [x[4] for x in used])
        rms = float(np.sqrt(np.mean([x[0].rms_px ** 2 for x in used])))
    return WorldEstimate(
        name=name, T_world_obj=T, capture_ns=max(x[2] for x in used), cameras=[x[0].camera for x in used],
        n_tags=sum(len(x[0].tag_ids) for x in used), rms_px=rms,
        weight=float(sum(x[4] for x in used)), ambiguous=len(used) == 1 and used[0][0].ambiguous,
    )


class StaticFilter:
    """Averages an object's pose while it stays still, resets as soon as it moves."""

    def __init__(self, window, still_pos_m, still_rot_deg):
        self.window = int(window)
        self.still_pos = float(still_pos_m)
        self.still_rot = float(still_rot_deg)
        self._buf = {}

    def update(self, est):
        """Returns the estimate, replaced by the running mean when the object is still."""
        buf = self._buf.setdefault(est.name, deque(maxlen=self.window))
        if buf:
            ref = average_poses([b[0] for b in buf], [b[1] for b in buf])
            dp, dr = pose_error(est.T_world_obj, ref)
            if dp > self.still_pos or dr > self.still_rot:
                buf.clear()
        buf.append((est.T_world_obj, est.weight))
        if len(buf) > 1:
            est.T_world_obj = average_poses([b[0] for b in buf], [b[1] for b in buf])
            est.static = True
            est.n_avg = len(buf)
        return est

    def reset(self, name=None):
        """Drops the history of one object, or of all objects."""
        if name is None:
            self._buf.clear()
        else:
            self._buf.pop(name, None)
