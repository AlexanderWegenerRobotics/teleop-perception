import time

import cv2
import numpy as np

from .config import Intrinsics
from .logger import RunLogger
from .net import GroundTruthListener, PosePublisher
from .pipeline import Pipeline
from .sources import make_source


def draw(gray, result, cam):
    """Debug overlay: tag outlines, ids and object axes."""
    img = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
    for i, c in result.detections.items():
        cv2.polylines(img, [c.astype(np.int32).reshape(-1, 1, 2)], True, (0, 255, 0), 1)
        cv2.putText(img, str(i), tuple(c[0].astype(int)), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 255), 1)
    for e in result.estimates:
        rvec, _ = cv2.Rodrigues(e.T_cam_obj[:3, :3])
        cv2.drawFrameAxes(img, cam.intrinsics.K, cam.intrinsics.dist, rvec, e.T_cam_obj[:3, 3], 0.03, 2)
    return img


class Runner:
    """Live loop: poll every camera, estimate, fuse, publish and log."""

    def __init__(self, cfg, show=False):
        self.cfg = cfg
        self.show = show
        self.cams = {c.name: c for c in cfg.cameras}
        self.sources = {c.name: make_source(c) for c in cfg.cameras}
        for name, src in self.sources.items():
            cam = self.cams[name]
            if cam.intrinsics_from_device:
                w, h, K, dist = src.intrinsics()
                cam.intrinsics = Intrinsics(width=w, height=h, K=K, dist=dist)
        self.pipeline = Pipeline(cfg)
        rt = cfg.detection.get("runtime", {})
        self.stale_ns = int(float(rt.get("stale_after_ms", 500)) * 1e6)
        self.idle_s = float(rt.get("idle_sleep_ms", 2)) / 1000.0
        self.sync_ns = int(float(cfg.detection.get("fusion", {}).get("sync_window_ms", 40)) * 1e6)
        net = cfg.network
        pub = net.get("publish", {})
        self.publisher = PosePublisher(pub["host"], pub["port"]) if pub.get("enabled", True) else None
        gt = net.get("ground_truth", {})
        self.gt = None
        self.gt_dt = int(float(gt.get("max_dt_ms", 30)) * 1e6)
        if cfg.mode == "sim" and gt.get("enabled", False):
            self.gt = GroundTruthListener(gt.get("bind_host", "0.0.0.0"), gt["port"])
        self.logger = RunLogger(cfg.logging["dir"], cfg) if cfg.logging.get("enabled", True) else None
        self.latest = {}
        self.last_frame_wall = {n: None for n in self.cams}
        self.state = {n: "waiting" for n in self.cams}
        self.t_start = time.time_ns()

    def _status(self, name, new_state):
        """Prints camera state changes once."""
        if self.state[name] != new_state:
            self.state[name] = new_state
            print(f"[perception] {name}: {new_state}")

    def run(self):
        """Runs until Ctrl+C."""
        for c in self.cams.values():
            print(f"[perception] {c.name}: extrinsics from {c.extrinsics_origin}")
        print(f"[perception] reference tags: {self.cfg.reference.get('origin')} ({len(self.cfg.reference_tags)})")
        try:
            while True:
                if not self.tick():
                    time.sleep(self.idle_s)
        except KeyboardInterrupt:
            pass
        finally:
            self.close()

    def tick(self):
        """One poll of all cameras; returns True if any new frame was processed."""
        now = time.time_ns()
        new = False
        for name, src in self.sources.items():
            f = src.read()
            if f is None:
                last = self.last_frame_wall[name]
                if last is None and now - self.t_start > 3e9:
                    self._status(name, f"no frames yet from {self.cams[name].source}")
                if last is not None and now - last > self.stale_ns:
                    self._status(name, "stale, dropped from fusion")
                    self.latest.pop(name, None)
                continue
            self.last_frame_wall[name] = now
            self._status(name, "live")
            res = self.pipeline.process_camera(self.cams[name], f.gray, f.capture_ns)
            self.latest[name] = res
            new = True
            if self.logger:
                self.logger.log_camera(res)
            if res.drift is not None and not res.drift[2]:
                print(f"[perception] {name}: reference tags off by {res.drift[0]:.2f} mm ({res.drift[1]:.2f} px)"
                      f" -- camera moved? recalibrate")
            if self.show:
                cv2.imshow(name, draw(f.gray, res, self.cams[name]))
        if not new:
            return False
        newest = max(r.capture_ns for r in self.latest.values())
        window = [r for r in self.latest.values() if newest - r.capture_ns <= self.sync_ns]
        fused = self.pipeline.fuse(window, self.cams)
        self.pipeline.n += 1
        gt = self.gt.lookup(newest, self.gt_dt) if self.gt else {}
        if self.publisher and fused:
            self.publisher.send(fused, newest)
        if self.logger:
            for r in window:
                for e in r.estimates:
                    self.logger.log_camera_estimate(r, e, self.cams[r.camera].T_world_cam, gt.get(e.name))
            for w in fused:
                self.logger.log_world(w, gt.get(w.name))
            if self.pipeline.n % 30 == 0:
                self.logger.flush()
        if self.show:
            cv2.waitKey(1)
        return True

    def close(self):
        """Releases sources, sockets and files."""
        for s in self.sources.values():
            s.close()
        if self.publisher:
            self.publisher.close()
        if self.gt:
            self.gt.close()
        if self.logger:
            self.logger.close()
            print(f"[perception] log: {self.logger.dir}")
        if self.show:
            cv2.destroyAllWindows()
