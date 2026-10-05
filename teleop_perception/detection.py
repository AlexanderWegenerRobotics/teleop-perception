from dataclasses import dataclass, field

import cv2
import numpy as np

from .geometry import T_from_rvec, inv_T, rvec_from_T

_REFINE = {
    "none": cv2.aruco.CORNER_REFINE_NONE,
    "subpix": cv2.aruco.CORNER_REFINE_SUBPIX,
    "contour": cv2.aruco.CORNER_REFINE_CONTOUR,
    "apriltag": cv2.aruco.CORNER_REFINE_APRILTAG,
}


class TagDetector:
    """ArUco/AprilTag detector with the parameters from detection.yaml."""

    def __init__(self, dictionary, params):
        self.dictionary = cv2.aruco.getPredefinedDictionary(getattr(cv2.aruco, dictionary))
        p = cv2.aruco.DetectorParameters()
        p.cornerRefinementMethod = _REFINE[params.get("corner_refinement", "subpix")]
        p.cornerRefinementWinSize = int(params.get("corner_refinement_win", 5))
        p.cornerRefinementMaxIterations = int(params.get("corner_refinement_max_iter", 50))
        p.cornerRefinementMinAccuracy = float(params.get("corner_refinement_min_accuracy", 0.01))
        p.adaptiveThreshWinSizeMin = int(params.get("adaptive_thresh_win_min", 3))
        p.adaptiveThreshWinSizeMax = int(params.get("adaptive_thresh_win_max", 23))
        p.adaptiveThreshWinSizeStep = int(params.get("adaptive_thresh_win_step", 10))
        p.minMarkerPerimeterRate = float(params.get("min_marker_perimeter_rate", 0.03))
        self.detector = cv2.aruco.ArucoDetector(self.dictionary, p)
        self.offset = float(params.get("corner_offset_px", 0.0))

    def detect(self, gray):
        """Returns {tag id: (4, 2) corner array in TL, TR, BR, BL order}."""
        corners, ids, _ = self.detector.detectMarkers(gray)
        if ids is None:
            return {}
        out = {}
        for c, i in zip(corners, ids.ravel()):
            i = int(i)
            if i in out:
                out.pop(i)
                continue
            out[i] = c.reshape(4, 2).astype(np.float64) + self.offset
        return out


@dataclass
class Estimate:
    name: str
    camera: str
    T_cam_obj: np.ndarray
    tag_ids: list
    n_corners: int
    rms_px: float
    view_angles_deg: list
    ambiguous: bool
    obj_pts: np.ndarray = None
    img_pts: np.ndarray = None
    rejected: dict = field(default_factory=dict)


def _side_px(c):
    """Shortest edge of a detected quad, pixels."""
    return float(min(np.linalg.norm(c[i] - c[(i + 1) % 4]) for i in range(4)))


def _reproj(T_cam_obj, obj_pts, img_pts, K, dist):
    """Per-point reprojection error of object points under a pose."""
    rvec, tvec = rvec_from_T(T_cam_obj)
    proj, _ = cv2.projectPoints(obj_pts, rvec, tvec, K, dist)
    return np.linalg.norm(proj.reshape(-1, 2) - img_pts, axis=1)


def _stack(tags, ids, detections):
    """Object points and image points for the given tag ids."""
    obj = np.vstack([tags[i].corners() for i in ids])
    img = np.vstack([detections[i] for i in ids])
    return obj, img


def _view_angle(T_cam_obj, tag):
    """Angle between a tag's normal and the ray from the tag to the camera, degrees."""
    T_cam_tag = T_cam_obj @ tag.T
    n = T_cam_tag[:3, 2]
    ray = -T_cam_tag[:3, 3] / np.linalg.norm(T_cam_tag[:3, 3])
    return float(np.degrees(np.arccos(np.clip(n @ ray, -1.0, 1.0))))


def _initial_pose(tags, ids, detections, K, dist):
    """Best IPPE solution of the largest few tags, judged on all visible corners."""
    obj_all, img_all = _stack(tags, ids, detections)
    best, best_err, errs = None, np.inf, []
    order = sorted(ids, key=lambda i: -cv2.contourArea(detections[i].astype(np.float32)))
    for i in order[:3]:
        h = tags[i].size / 2.0
        local = np.array([[-h, h, 0.0], [h, h, 0.0], [h, -h, 0.0], [-h, -h, 0.0]])
        n, rvecs, tvecs, _ = cv2.solvePnPGeneric(local, detections[i], K, dist, flags=cv2.SOLVEPNP_IPPE_SQUARE)
        for r, t in zip(rvecs[:n], tvecs[:n]):
            T_cam_obj = T_from_rvec(r, t) @ inv_T(tags[i].T)
            if T_cam_obj[2, 3] <= 0.0:
                continue
            e = float(np.sqrt(np.mean(_reproj(T_cam_obj, obj_all, img_all, K, dist) ** 2)))
            errs.append(e)
            if e < best_err:
                best, best_err = T_cam_obj, e
    errs.sort()
    ambiguous = len(ids) == 1 and len(errs) > 1 and errs[1] < max(3.0 * errs[0], 0.3)
    return best, ambiguous


def _refine(T_cam_obj, tags, ids, detections, K, dist):
    """Levenberg-Marquardt refinement on all corners of the given tags."""
    obj, img = _stack(tags, ids, detections)
    rvec, tvec = rvec_from_T(T_cam_obj)
    rvec, tvec = cv2.solvePnPRefineLM(obj, img, K, dist, rvec, tvec)
    T = T_from_rvec(rvec, tvec)
    err = _reproj(T, obj, img, K, dist)
    return T, err


def estimate_object(name, camera, tags, detections, K, dist, params):
    """Pose of one rigid tag group from every visible tag at once, or None."""
    min_side = float(params.get("min_side_px", 12))
    max_angle = float(params.get("max_view_angle_deg", 65.0))
    max_tag_err = float(params.get("max_tag_reproj_px", 2.0))
    min_tags = int(params.get("min_tags", 1))
    rejected = {}
    ids = []
    for i in tags:
        if i not in detections:
            continue
        if _side_px(detections[i]) < min_side:
            rejected[i] = "small"
            continue
        ids.append(i)
    if len(ids) < min_tags:
        return None
    T, ambiguous = _initial_pose(tags, ids, detections, K, dist)
    if T is None:
        return None
    for _ in range(3):
        T, err = _refine(T, tags, ids, detections, K, dist)
        keep = []
        for k, i in enumerate(ids):
            angle = _view_angle(T, tags[i])
            tag_err = float(np.sqrt(np.mean(err[4 * k:4 * k + 4] ** 2)))
            if angle > max_angle:
                rejected[i] = "angle"
            elif tag_err > max_tag_err and len(ids) > 1:
                rejected[i] = "reproj"
            else:
                keep.append(i)
        if len(keep) == len(ids):
            break
        if len(keep) < min_tags:
            return None
        ids = keep
    T, err = _refine(T, tags, ids, detections, K, dist)
    obj, img = _stack(tags, ids, detections)
    return Estimate(
        name=name, camera=camera, T_cam_obj=T, tag_ids=list(ids), n_corners=4 * len(ids),
        rms_px=float(np.sqrt(np.mean(err ** 2))), view_angles_deg=[_view_angle(T, tags[i]) for i in ids],
        ambiguous=bool(ambiguous and len(ids) == 1), obj_pts=obj, img_pts=img, rejected=rejected,
    )
