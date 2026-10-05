import csv
import os
import time

import yaml

from .geometry import pose_error, quat_from_R

POSE_COLUMNS = [
    "capture_ns", "process_ns", "source", "object", "x", "y", "z", "qw", "qx", "qy", "qz",
    "n_tags", "n_cams", "rms_px", "static", "n_avg", "ambiguous",
    "gt_x", "gt_y", "gt_z", "gt_qw", "gt_qx", "gt_qy", "gt_qz", "err_pos_mm", "err_rot_deg",
]
CAMERA_COLUMNS = ["capture_ns", "camera", "n_detected", "detected_ids", "drift_mm", "drift_px", "drift_ok", "notes"]


class RunLogger:
    """Writes poses.csv and cameras.csv for one run, plus a copy of the resolved setup."""

    def __init__(self, root, cfg):
        stamp = time.strftime("%Y%m%d_%H%M%S")
        self.dir = os.path.join(root, f"{stamp}_{cfg.mode}_{cfg.board.name}")
        os.makedirs(self.dir, exist_ok=True)
        self._pf = open(os.path.join(self.dir, "poses.csv"), "w", newline="")
        self._cf = open(os.path.join(self.dir, "cameras.csv"), "w", newline="")
        self.poses = csv.writer(self._pf, delimiter=";")
        self.cams = csv.writer(self._cf, delimiter=";")
        self.poses.writerow(POSE_COLUMNS)
        self.cams.writerow(CAMERA_COLUMNS)
        meta = {
            "config": cfg.path, "mode": cfg.mode, "board": cfg.board.name,
            "cameras": {c.name: {"extrinsics_origin": c.extrinsics_origin,
                                 "T_world_cam": None if c.T_world_cam is None else c.T_world_cam.tolist()}
                        for c in cfg.cameras},
            "reference_origin": cfg.reference.get("origin"),
        }
        with open(os.path.join(self.dir, "meta.yaml"), "w") as f:
            yaml.safe_dump(meta, f, sort_keys=False)

    def log_pose(self, capture_ns, source, name, T, n_tags, n_cams, rms_px, static=False, n_avg=1,
                 ambiguous=False, gt=None):
        """Writes one pose row, with ground truth and error if available."""
        q = quat_from_R(T[:3, :3])
        row = [capture_ns, time.time_ns(), source, name, *[f"{v:.6f}" for v in T[:3, 3]],
               *[f"{v:.6f}" for v in q], n_tags, n_cams, f"{rms_px:.3f}", int(static), n_avg, int(ambiguous)]
        if gt is not None:
            dp, dr = pose_error(T, gt)
            row += [*[f"{v:.6f}" for v in gt[:3, 3]], *[f"{v:.6f}" for v in quat_from_R(gt[:3, :3])],
                    f"{dp * 1000:.3f}", f"{dr:.3f}"]
        else:
            row += [""] * 9
        self.poses.writerow(row)

    def log_world(self, w, gt=None):
        """Writes a fused WorldEstimate."""
        self.log_pose(w.capture_ns, "fused", w.name, w.T_world_obj, w.n_tags, len(w.cameras), w.rms_px,
                      w.static, w.n_avg, w.ambiguous, gt)

    def log_camera_estimate(self, r, e, T_world_cam, gt=None):
        """Writes one camera's unfused estimate in world frame."""
        self.log_pose(r.capture_ns, r.camera, e.name, T_world_cam @ e.T_cam_obj, len(e.tag_ids), 1, e.rms_px,
                      ambiguous=e.ambiguous, gt=gt)

    def log_camera(self, r):
        """Writes one camera row."""
        d = r.drift or ("", "", "")
        self.cams.writerow([r.capture_ns, r.camera, len(r.detections), " ".join(map(str, sorted(r.detections))),
                            f"{d[0]:.3f}" if d[0] != "" else "", f"{d[1]:.3f}" if d[1] != "" else "",
                            int(d[2]) if d[2] != "" else "", " | ".join(r.notes)])

    def flush(self):
        """Flushes both files."""
        self._pf.flush()
        self._cf.flush()

    def close(self):
        """Closes both files."""
        self._pf.close()
        self._cf.close()
