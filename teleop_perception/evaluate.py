import csv
import os

import numpy as np


def summarize(run_dir):
    """Prints per-object error statistics of a logged run against the logged ground truth."""
    rows = []
    with open(os.path.join(run_dir, "poses.csv"), newline="") as f:
        for r in csv.DictReader(f, delimiter=";"):
            rows.append(r)
    if not rows:
        print("no poses logged")
        return
    t = np.array([int(r["capture_ns"]) for r in rows])
    dur = (t.max() - t.min()) / 1e9
    print(f"{run_dir}: {len(rows)} rows over {dur:.1f} s")
    print(f"{'object':10s} {'source':18s} {'rows':>6s} {'with gt':>8s} {'pos med':>9s} {'pos p95':>9s} "
          f"{'pos max':>9s} {'rot med':>8s} {'rot p95':>8s} {'static':>7s}")
    keys = sorted({(r["object"], r["source"]) for r in rows}, key=lambda k: (k[0], k[1] == "fused", k[1]))
    for obj, src in keys:
        rs = [r for r in rows if r["object"] == obj and r["source"] == src]
        e = np.array([float(r["err_pos_mm"]) for r in rs if r["err_pos_mm"]])
        a = np.array([float(r["err_rot_deg"]) for r in rs if r["err_rot_deg"]])
        st = np.mean([int(r["static"]) for r in rs])
        if len(e):
            print(f"{obj:10s} {src:18s} {len(rs):6d} {len(e):8d} {np.median(e):7.2f}mm {np.percentile(e, 95):7.2f}mm "
                  f"{e.max():7.2f}mm {np.median(a):7.2f}d {np.percentile(a, 95):7.2f}d {st:7.2f}")
        else:
            print(f"{obj:10s} {src:18s} {len(rs):6d} {0:8d}")
    cams = os.path.join(run_dir, "cameras.csv")
    if os.path.exists(cams):
        with open(cams, newline="") as f:
            cr = list(csv.DictReader(f, delimiter=";"))
        for cam in sorted({r["camera"] for r in cr}):
            c = [r for r in cr if r["camera"] == cam]
            d = [float(r["drift_mm"]) for r in c if r["drift_mm"]]
            ts = np.array([int(r["capture_ns"]) for r in c])
            fps = (len(ts) - 1) / ((ts.max() - ts.min()) / 1e9) if len(ts) > 1 and ts.max() > ts.min() else 0.0
            drift = f", reference check median {np.median(d):.3f} mm (max {max(d):.3f})" if d else ""
            print(f"{cam}: {len(c)} frames, {fps:.1f} fps, {np.mean([int(r['n_detected']) for r in c]):.1f} tags/frame"
                  f"{drift}")
