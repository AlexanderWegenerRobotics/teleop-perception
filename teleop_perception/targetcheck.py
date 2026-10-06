import time

import numpy as np

from .calibrate import _for_role
from .sources import make_source
from .targets import make_target


def marker_px(target, pts):
    """Median marker side in pixels from the spacing of neighbouring detected corners."""
    nx = target.board.getChessboardSize()[0] - 1
    d = []
    for i in range(len(pts) - 1):
        if (i + 1) % nx and not np.isnan(pts[i, 0]) and not np.isnan(pts[i + 1, 0]):
            d.append(np.linalg.norm(pts[i + 1] - pts[i]))
    return float(np.median(d)) * target.marker / target.square if d else 0.0


def run(cfg, role="perception", period=1.0):
    """Prints, once per period, how many target corners each camera sees and how large the markers are."""
    cfg = _for_role(cfg, role)
    target = make_target(cfg)
    sources = {c.name: make_source(c) for c in cfg.cameras}
    print(f"[check] {target.kind}: {len(target.points)} corners; markers should be >= ~25 px for blurred real images")
    try:
        while True:
            t_end = time.monotonic() + period
            for name, src in sources.items():
                f = None
                while time.monotonic() < t_end and f is None:
                    f = src.read()
                    time.sleep(0.005)
                if f is None:
                    print(f"[check] {name}: no frame")
                    continue
                pts = target.detect(f.gray)
                n = int(np.sum(~np.isnan(pts[:, 0])))
                size = f" marker {marker_px(target, pts):.0f} px" if getattr(target, "board", None) is not None and n else ""
                print(f"[check] {name}: {n}/{len(target.points)} corners{size}")
            time.sleep(max(0.0, t_end - time.monotonic()))
    except KeyboardInterrupt:
        pass
    finally:
        for s in sources.values():
            s.close()
