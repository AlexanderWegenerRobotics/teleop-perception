from dataclasses import dataclass, field

from .detection import TagDetector, estimate_object
from .fusion import StaticFilter, fuse
from .reference import DriftMonitor, camera_from_reference


@dataclass
class CameraResult:
    camera: str
    capture_ns: int
    detections: dict
    estimates: list
    drift: tuple = None
    notes: list = field(default_factory=list)


class Pipeline:
    """Detection, per-object pose, multi-camera fusion and still-averaging for one set of frames."""

    def __init__(self, cfg):
        self.cfg = cfg
        det = cfg.detection
        self.detector = TagDetector(cfg.board.dictionary, det.get("detector", {}))
        self.est_params = det.get("estimation", {})
        self.fusion_params = det.get("fusion", {})
        sm = det.get("smoothing", {})
        self.filter = StaticFilter(sm.get("window", 15), sm.get("still_pos_m", 0.0015), sm.get("still_rot_deg", 1.0)) \
            if sm.get("enabled", True) else None
        ref = cfg.reference
        self.ref_mode = ref.get("mode", "check")
        self.ref_every = int(ref.get("check_every_n", 30))
        self.drift = DriftMonitor(cfg.reference_tags, ref.get("drift_tol_m", 0.001), ref.get("observations"))
        self.models = {o.name: o.tags for o in cfg.board.objects}
        self.n = 0

    def process_camera(self, cam, gray, capture_ns):
        """Detects tags in one image and estimates every object the camera sees."""
        intr = cam.intrinsics
        dets = self.detector.detect(gray)
        res = CameraResult(camera=cam.name, capture_ns=capture_ns, detections=dets, estimates=[])
        if self.cfg.reference_tags and self.ref_mode != "off":
            if cam.T_world_cam is None or self.ref_mode == "track":
                sol = camera_from_reference(self.cfg.reference_tags, cam.name, dets, intr.K, intr.dist, self.est_params)
                if sol is not None:
                    if cam.T_world_cam is None:
                        res.notes.append("extrinsics bootstrapped from reference tags")
                    cam.T_world_cam, cam.extrinsics_origin = sol[0], "reference"
            elif self.n % self.ref_every == 0:
                res.drift = self.drift.check(cam.name, cam.T_world_cam, dets, intr.K, intr.dist)
        if cam.T_world_cam is None:
            res.notes.append("no extrinsics")
            return res
        for name, tags in self.models.items():
            e = estimate_object(name, cam.name, tags, dets, intr.K, intr.dist, self.est_params)
            if e is not None:
                res.estimates.append(e)
        return res

    def fuse(self, results, cams):
        """Fuses per-camera estimates into one world pose per object."""
        per_obj = {}
        for r in results:
            cam = cams[r.camera]
            if cam.T_world_cam is None:
                continue
            for e in r.estimates:
                per_obj.setdefault(e.name, []).append((e, cam.T_world_cam @ e.T_cam_obj, r.capture_ns, cam))
        out = []
        for name, items in per_obj.items():
            w = fuse(name, items, self.fusion_params)
            if self.filter is not None:
                w = self.filter.update(w)
            out.append(w)
        return out

    def step(self, frames, cams):
        """Processes {camera: (gray, capture_ns)} and returns (world estimates, camera results)."""
        results = [self.process_camera(cams[c], g, t) for c, (g, t) in frames.items()]
        self.n += 1
        return self.fuse(results, cams), results
