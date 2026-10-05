import cv2
import numpy as np

from .detection import estimate_object
from .geometry import inv_T, rvec_from_T


def camera_from_reference(ref_tags, camera, detections, K, dist, params):
    """World-from-camera pose solved from the visible reference tags, or None."""
    tags = {t.id: t for t in ref_tags}
    p = dict(params)
    p["min_tags"] = max(2, int(p.get("min_tags", 1)))
    est = estimate_object("reference", camera, tags, detections, K, dist, p)
    if est is None:
        return None
    return inv_T(est.T_cam_obj), est


def workspace_drift(T_world_cam, T_world_cam_ref, points_world):
    """Mean displacement of workspace points when seen through T_world_cam instead of T_world_cam_ref."""
    P = np.c_[np.asarray(points_world), np.ones(len(points_world))]
    moved = (T_world_cam @ inv_T(T_world_cam_ref) @ P.T).T[:, :3]
    return float(np.mean(np.linalg.norm(moved - P[:, :3], axis=1)))


def reference_reprojection(ref_tags, detections, T_world_cam, K, dist):
    """Median corner error of the visible reference tags under stored extrinsics, in px and as mm at the tags."""
    T_cw = inv_T(T_world_cam)
    rv, tv = rvec_from_T(T_cw)
    px, mm = [], []
    for t in ref_tags:
        if t.id not in detections:
            continue
        pts = t.corners()
        proj, _ = cv2.projectPoints(pts, rv, tv, K, dist)
        e = np.linalg.norm(proj.reshape(-1, 2) - detections[t.id], axis=1)
        depth = (T_cw[:3, :3] @ pts.T + T_cw[:3, 3:4])[2]
        px.append(e)
        mm.append(e * depth / K[1, 1] * 1000.0)
    if not px:
        return None
    return float(np.median(np.concatenate(px))), float(np.median(np.concatenate(mm))), len(px)


def observed_shift(observations, detections, K):
    """Median corner shift against the corners stored after calibration, in px and as mm at the tags."""
    px, mm = [], []
    for tid, ref in observations.items():
        if tid not in detections:
            continue
        e = np.linalg.norm(detections[tid] - ref["corners"], axis=1)
        px.append(e)
        mm.append(e * ref["depth_m"] / K[1, 1] * 1000.0)
    if not px:
        return None
    return float(np.median(np.concatenate(px))), float(np.median(np.concatenate(mm))), len(px)


class DriftMonitor:
    """Detects a moved camera: reference tags must stay where the camera saw them after calibration."""

    def __init__(self, ref_tags, tol_m, observations=None):
        self.ref_tags = ref_tags
        self.tol_mm = float(tol_m) * 1000.0
        self.observations = observations or {}
        self.last = {}

    def check(self, camera, T_world_cam, detections, K, dist):
        """Returns (drift mm, drift px, within tolerance) or None if no reference tag is visible."""
        if camera in self.observations:
            r = observed_shift(self.observations[camera], detections, K)
        else:
            r = reference_reprojection(self.ref_tags, detections, T_world_cam, K, dist)
        if r is None:
            return None
        px, mm, _ = r
        out = (mm, px, mm <= self.tol_mm)
        self.last[camera] = out
        return out
