import dataclasses
import os
import time
import warnings

import cv2
import numpy as np
import yaml

from . import wire
from .config import Intrinsics, calib_dir
from .detection import TagDetector
from .geometry import T_from_rvec, make_T, pose_error, to_list
from .overlay import overlay_K, overlay_view, pixel_error, project, workspace_points, write_overlay
from .kinematics import Arm
from .planner import candidates, motion_cost, order_poses
from .reference import workspace_drift
from .solve import cell_hits, coverage_map, model_shift_px, solve
from .sources import make_source
from .targets import make_target


def tag_square(size):
    """Corners of a square tag in its own frame, OpenCV order."""
    h = size / 2.0
    return np.array([[-h, h, 0.0], [h, h, 0.0], [h, -h, 0.0], [-h, -h, 0.0]])


def generate_poses(T0, motion, rng, model=None, q0=None):
    """Random poses around T0 inside the workspace box and reachable from q0 within the joint limits, short travel first."""
    rng_pos = np.asarray(motion.get("pos_range_m", [0.08, 0.08, 0.06]), dtype=float)
    rot = np.radians(float(motion.get("rot_range_deg", 25.0)))
    lo = np.asarray(motion.get("workspace_min", [-np.inf] * 3), dtype=float)
    hi = np.asarray(motion.get("workspace_max", [np.inf] * 3), dtype=float)
    poses, n = [], int(motion.get("n_poses", 18))
    for _ in range(50 * n):
        if len(poses) == n:
            break
        p = np.clip(T0[:3, 3] + rng.uniform(-rng_pos, rng_pos), lo, hi)
        axis = rng.normal(size=3)
        axis /= np.linalg.norm(axis)
        R, _ = cv2.Rodrigues(axis * rng.uniform(0.3, 1.0) * rot)
        P = make_T(T0[:3, :3] @ R, p)
        if model is None or model.follow(q0, P) is not None:
            poses.append(P)
    return order_poses(poses, T0)


def _ippe(corners, size, K, dist):
    """Lower-error IPPE pose of a single square tag."""
    n, rvecs, tvecs, errs = cv2.solvePnPGeneric(tag_square(size), corners, K, dist, flags=cv2.SOLVEPNP_IPPE_SQUARE)
    k = int(np.argmin(np.ravel(errs)[:n]))
    return T_from_rvec(rvecs[k], tvecs[k])


class ArmNotFollowing(RuntimeError):
    """The arm ignored the command stream: it faulted, or the avatar dropped POLICY to HOLD."""


def _arm_model(cfg):
    """Kinematic model for checking moves against the joint limits, or None if not configured."""
    spec = cfg.calibration.get("kinematics")
    return Arm(spec) if spec else None


def _q(arm):
    """Measured joint positions; raises if the avatar does not send them."""
    q = np.asarray(arm.latest()[2]["joints"], dtype=float)
    if not np.any(q):
        raise ArmNotFollowing("the arm state has no joint positions (all zero), so moves cannot be checked against "
                              "the joint limits. Rebuild the avatar with joint_positions filled in arm_control.cpp.")
    return q


def _feasible(model, arm, T):
    """True if the straight move from the current joints to T keeps every joint off its limits."""
    return model is None or model.follow(_q(arm), T) is not None


def _check_fault(arm):
    """Raises if the arm left ENGAGED or reports a fault."""
    s = arm.latest()[2]
    if s["fault"] or s["state"] != wire.ENGAGED:
        raise ArmNotFollowing(f"arm is not ENGAGED (state {s['state']}, fault code {s['fault']}). Reset it "
                              f"(arm_reset) and look for FRANKA ERROR in avatar_stdout.log.")


def _settle(arm, T, st):
    """Moves to T and waits until the arm is still there; raises if the arm did not move at all."""
    T_start = arm.latest()[1]
    arm.pop_stream_stats()
    arm.move_to(T)
    T_meas = arm.wait_settled(st.get("window_s", 0.5), st.get("pos_tol_m", 2e-4), st.get("rot_tol_deg", 0.05),
                              st.get("timeout_s", 6.0), goal=T,
                              goal_tol=(st.get("goal_tol_m", 0.02), st.get("goal_tol_deg", 3.0)))
    if T_meas is not None:
        return T_meas
    _check_fault(arm)
    gap, errors, err_msg = arm.pop_stream_stats()
    asked_p, asked_r = pose_error(T_start, T)
    moved_p, moved_r = pose_error(T_start, arm.latest()[1])
    if (asked_p > 0.01 or asked_r > 5.0) and moved_p < 0.002 and moved_r < 1.0:
        detail = f"largest gap between commands {gap * 1000:.0f} ms"
        if errors:
            detail += f", {errors} send errors ({err_msg})"
        raise ArmNotFollowing(f"arm stopped following the commands ({detail}). The avatar probably switched "
                              f"POLICY to HOLD -- look for 'no command for' in avatar_stdout.log.")
    print(f"[calib] arm did not settle at the target (moved {moved_p * 1000:.0f} mm / {moved_r:.1f} deg "
          f"of {asked_p * 1000:.0f} mm / {asked_r:.1f} deg)")
    return None


def face_cameras(arm, sources, target, T0, motion, st, model=None):
    """Moves to the start position and turns the hand about its tool axis until the target faces the cameras."""
    T = T0.copy()
    if motion.get("start_pos") is not None:
        T[:3, 3] = np.asarray(motion["start_pos"], dtype=float)
    best, best_n = None, 0
    for yaw in motion.get("search_yaw_deg", [0, 45, -45, 90, -90, 135, -135, 180]):
        R, _ = cv2.Rodrigues(np.array([0.0, 0.0, np.radians(yaw)]))
        cand = make_T(T[:3, :3] @ R, T[:3, 3])
        if not _feasible(model, arm, cand):
            print(f"[calib] start yaw {yaw:+d} deg: too close to a joint limit, skipped")
            continue
        if _settle(arm, cand, st) is None:
            print(f"[calib] start yaw {yaw:+d} deg: not reached, skipped")
            continue
        seen = sum(c is not None for c in _capture(sources, target, 5).values())
        print(f"[calib] start yaw {yaw:+d} deg: target seen by {seen}/{len(sources)} cameras")
        if seen > best_n:
            best, best_n = cand, seen
        if seen == len(sources):
            break
    if best is None:
        raise RuntimeError("calibration target not visible from any start yaw -- check the mount and start_pos")
    return best


def _capture(sources, target, n_frames, timeout=5.0):
    """Collects n_frames per camera; returns {camera: mean image points (NaN where unseen) or None}."""
    got = {c: [] for c in sources}
    seen = {c: 0 for c in sources}
    t_end = time.monotonic() + timeout
    while time.monotonic() < t_end and any(seen[c] < n_frames for c in sources):
        for name, src in sources.items():
            if seen[name] >= n_frames:
                continue
            f = src.read()
            if f is None:
                continue
            seen[name] += 1
            got[name].append(target.detect(f.gray))
        time.sleep(0.002)
    out = {}
    for c, v in got.items():
        if not v:
            out[c] = None
            continue
        A = np.array(v)
        n = np.sum(~np.isnan(A[:, :, 0]), axis=0)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            pts = np.nanmean(A, axis=0)
        pts[n < max(1, n_frames // 2)] = np.nan
        out[c] = pts if np.sum(~np.isnan(pts[:, 0])) >= target.min_points else None
    return out


def measure_reference(sources, cams, detector, ref_tags, n_frames):
    """Where each camera sees each reference tag right after calibration: median corners and depth per tag."""
    ids = {t.id: t for t in ref_tags}
    seen = {c: {} for c in sources}
    count = {c: 0 for c in sources}
    t_end = time.monotonic() + 20.0
    while time.monotonic() < t_end and any(v < n_frames for v in count.values()):
        for name, src in sources.items():
            f = src.read()
            if f is None or count[name] >= n_frames:
                continue
            count[name] += 1
            for tid, c in detector.detect(f.gray).items():
                if tid in ids:
                    seen[name].setdefault(tid, []).append(c)
        time.sleep(0.002)
    for name, n in count.items():
        if n < n_frames:
            print(f"[calib] {name}: only {n}/{n_frames} frames for the reference tags -- is the sim running?")
    out = {}
    for name, tags in seen.items():
        intr = cams[name].intrinsics
        for tid, cs in tags.items():
            if len(cs) < max(3, n_frames // 2):
                continue
            corners = np.median(np.array(cs), axis=0)
            depth = float(_ippe(corners, ids[tid].size, intr.K, intr.dist)[2, 3])
            out.setdefault(name, {})[int(tid)] = {"corners": [[float(u), float(v)] for u, v in corners],
                                                  "depth_m": depth}
    return out


def _visit_one(arm, P, sources, target, samples, st, cap):
    """Settles at P and captures the target in every camera; returns the cameras that saw it, or None if not reached."""
    if _settle(arm, P, st) is None:
        return None
    t0 = time.monotonic()
    pts = _capture(sources, target, int(cap.get("frames", 15)))
    T_mean = arm.measured_since(t0)
    T_mean = arm.latest()[1] if T_mean is None else T_mean
    seen = [n for n, p in pts.items() if p is not None]
    for n in seen:
        samples[n].append((T_mean, pts[n]))
    return seen


def _visit(arm, poses, sources, target, samples, st, cap, label, model=None):
    """Visits a fixed list of poses, skipping moves that would bring a joint near its limit."""
    for k, P in enumerate(poses):
        if not _feasible(model, arm, P):
            print(f"[calib] {label} {k + 1}/{len(poses)}: too close to a joint limit from here, skipped")
            continue
        seen = _visit_one(arm, P, sources, target, samples, st, cap)
        msg = "arm did not settle, skipped" if seen is None else f"target seen by {seen or 'no camera'}"
        print(f"[calib] {label} {k + 1}/{len(poses)}: {msg}")


def _wants_intrinsics(cfg, target):
    """True if this run estimates intrinsics."""
    return target.supports_intrinsics and bool(cfg.calibration.get("intrinsics", {}).get("estimate", False))


def _print_map(name, hits, need):
    """Prints which image cells the target has covered."""
    print(f"[calib] {name} image coverage ('#' covered, '+' few points, '.' none):")
    for row in coverage_map(hits, need):
        print(f"          {row}")


def _next_pose(arm, model, todo, open_):
    """Cheapest (index, option) among the open cells whose move stays off the joint limits, or None."""
    cur = arm.latest()[1]
    q = None if model is None else _q(arm)
    best, best_cost = None, np.inf
    for k in open_:
        for o, P in enumerate(todo[k][1]):
            if model is not None and model.follow(q, P) is None:
                continue
            cost = motion_cost(cur, P)
            if cost < best_cost:
                best, best_cost = (k, o), cost
            break
    return best


def _cover(arm, cfg, cams, sources, target, samples, T0, st, cap, model=None):
    """Per camera and distance: keeps visiting the cheapest feasible pose for an image cell not yet covered at that distance."""
    cov = cfg.calibration.get("coverage", {})
    grid = tuple(int(v) for v in cov.get("grid", [5, 3]))
    need = int(cov.get("min_points_per_cell", 4))
    max_poses = int(cov.get("max_poses_per_camera", 30))
    for name, cam in cams.items():
        intr = cam.intrinsics
        size = (intr.width, intr.height)
        try:
            r = solve(samples[name], target.points, intr.K, intr.dist)
        except (RuntimeError, ValueError, cv2.error) as e:
            print(f"[calib] {name}: no coverage poses, first-stage solve failed ({e})")
            continue
        cand = candidates(r["T_world_cam"], intr.K, size, target.points, r["T_ee_target"], T0[:3, :3], cov)
        n_d = len(cov.get("distances_m", [0.5]))
        print(f"[calib] {name}: {len(cand)} of {grid[0] * grid[1] * n_d} cell/distance poses inside the workspace")
        n = 0
        for dist_i in range(n_d):
            own = []
            todo = [(c, list(opts)) for i, c, opts in cand if i == dist_i]
            recovered = False
            while todo and n < max_poses:
                hits = cell_hits(own, size, grid)
                open_ = [k for k, (c, opts) in enumerate(todo) if hits[c[1], c[0]] < need and opts]
                if not open_:
                    break
                pick = _next_pose(arm, model, todo, open_)
                if pick is None:
                    if recovered or not _feasible(model, arm, T0):
                        print(f"[calib] {name} distance {dist_i + 1}/{n_d}: no remaining cell reachable within the "
                              f"joint limits")
                        break
                    print(f"[calib] {name}: nothing reachable from here, returning to the start pose")
                    _settle(arm, T0, st)
                    recovered = True
                    continue
                recovered = False
                k, o = pick
                cell, opts = todo[k]
                P = opts.pop(o)
                n += 1
                before = len(samples[name])
                seen = _visit_one(arm, P, sources, target, samples, st, cap)
                if len(samples[name]) > before:
                    own.append(samples[name][-1])
                    opts.clear()
                msg = "not reached" if seen is None else ("seen" if name in seen else "not seen")
                print(f"[calib] {name} distance {dist_i + 1}/{n_d} cell {cell}: {msg}")
        _print_map(name, cell_hits(samples[name], size, grid), need)


def _collect(cfg, cams, sources, target):
    """Engages the arm, finds a start yaw, visits the calibration (and coverage) poses; returns samples or None."""
    c = cfg.calibration
    from .arm import ArmClient
    arm = ArmClient(c["arm"], c.get("motion", {}))
    samples = {name: [] for name in cams}
    engaged_here = False
    interrupted = False
    try:
        arm.wait_for_state()
        if c["arm"].get("engage", True):
            engaged_here = arm.engage()
            print(f"[calib] arm {'engaged (homed to q0)' if engaged_here else 'already ENGAGED'}")
        _, T0, _ = arm.wait_for_state()
        if c["arm"].get("request_authority", True):
            if not arm.request_authority(wire.POLICY):
                raise RuntimeError("avatar did not ack the authority request")
            print("[calib] POLICY authority requested -- release HUMAN in the interface if the arm does not move")
        arm.start_streaming()
        st = c.get("settle", {})
        cap = c.get("capture", {})
        T_home = T0
        model = _arm_model(cfg)
        T0 = face_cameras(arm, sources, target, T0, c.get("motion", {}), st, model)
        poses = generate_poses(T0, c.get("motion", {}), np.random.default_rng(int(c["motion"].get("seed", 0))),
                               model, None if model is None else _q(arm))
        _visit(arm, poses, sources, target, samples, st, cap, "pose", model)
        if _wants_intrinsics(cfg, target):
            _cover(arm, cfg, cams, sources, target, samples, T0, st, cap, model)
        arm.move_to(T_home)
        arm.wait_settled(0.5, 1e-3, 0.5, 10.0)
    except KeyboardInterrupt:
        print("\n[calib] interrupted -- releasing the arm, nothing written")
        interrupted = True
    except ArmNotFollowing as e:
        print(f"[calib] {e}")
        interrupted = True
    finally:
        if engaged_here:
            arm.disengage()
        arm.close(release=c["arm"].get("request_authority", True))
    if interrupted:
        return None
    return samples


def save_samples(path, samples, target):
    """Writes {camera: [(T_world_ee, image points)]} to an npz."""
    data = {"target": np.array(target.kind)}
    for n, v in samples.items():
        if v:
            data[f"{n}_T"] = np.array([x[0] for x in v])
            data[f"{n}_pts"] = np.array([x[1] for x in v])
    np.savez(path, **data)


def load_samples(path, cams, target):
    """Samples saved by an earlier calibration run, {camera: [(T_world_ee, image points)]}."""
    z = np.load(path)
    kind = str(z["target"]) if "target" in z else "gripper_tag"
    if kind != target.kind:
        raise ValueError(f"{path} was recorded with the {kind} target, calibration.yaml selects {target.kind}")
    out = {}
    for n in cams:
        key = f"{n}_pts" if f"{n}_pts" in z else f"{n}_corners"
        out[n] = list(zip(z[f"{n}_T"], z[key])) if f"{n}_T" in z else []
    return out


def _fmt_K(K, d):
    """One-line intrinsics summary."""
    return (f"fx {K[0, 0]:.2f} fy {K[1, 1]:.2f} cx {K[0, 2]:.2f} cy {K[1, 2]:.2f} "
            f"dist [{', '.join(f'{v:+.4f}' for v in np.ravel(d)[:5])}]")


def _report_intrinsics(name, r, base, base_label, size, cov):
    """Prints the estimated intrinsics, their uncertainty, the image coverage and the change against the base model."""
    sx = r["sigma_K"]
    print(f"[calib] {name}: intrinsics {_fmt_K(r['K'], r['dist'])}")
    print(f"[calib] {name}: 1-sigma fx {sx[0]:.2f} fy {sx[1]:.2f} cx {sx[2]:.2f} cy {sx[3]:.2f} px")
    grid = tuple(int(v) for v in cov.get("grid", [5, 3]))
    _print_map(name, cell_hits(r["samples"], size, grid), int(cov.get("min_points_per_cell", 4)))
    if base is not None:
        mx, mn = model_shift_px(base.K, base.dist, r["K"], r["dist"], size)
        print(f"[calib] {name}: vs {base_label} ({_fmt_K(base.K, base.dist)}): {mn:.2f} px mean, {mx:.2f} px max "
              f"over the image")


def _solve_camera(s, target, intr, est, dist_terms, min_valid, name):
    """Solves one camera and drops outlier poses once."""
    r = solve(s, target.points, intr.K, intr.dist, est, dist_terms)
    per = r["rms_px"]
    good = per <= max(1.0, 3.0 * float(np.median(per)))
    if not good.all() and good.sum() >= min_valid:
        print(f"[calib] {name}: dropping {int((~good).sum())} outlier poses")
        r = solve([x for x, g in zip(r["samples"], good) if g], target.points, intr.K, intr.dist, est, dist_terms)
    return r


def _merge(base, over):
    """Recursive dict merge, over wins."""
    out = dict(base)
    for k, v in over.items():
        out[k] = _merge(out[k], v) if isinstance(v, dict) and isinstance(out.get(k), dict) else v
    return out


def _for_role(cfg, role):
    """Config whose cameras and calibration settings are those of one camera role."""
    if role == "perception":
        return cfg
    cams = cfg.operator_cameras if role == "operator" else []
    if not cams:
        raise SystemExit(f"[calib] no cameras with role '{role}' in the camera file for mode '{cfg.mode}'")
    calib = _merge({k: v for k, v in cfg.calibration.items() if k != role}, cfg.calibration.get(role, {}))
    return dataclasses.replace(cfg, cameras=cams, calibration=calib)


def _save_reference(path, ref, stamp, mode):
    """Merges the newly measured reference observations into the reference file."""
    old = {}
    if os.path.exists(path):
        with open(path) as f:
            old = (yaml.safe_load(f) or {}).get("cameras", {})
    old.update(ref)
    with open(path, "w") as f:
        yaml.safe_dump({"date": stamp, "mode": mode, "cameras": old}, f, sort_keys=False)


def _report_overlay(cfg, name, cam, r, size, stamp, base):
    """Writes the VR overlay viewpoint of an operator camera and, in sim, its pixel error against the truth."""
    c = cfg.calibration
    view = overlay_view(r["T_world_cam"], r["K"], size)
    out = os.path.join(calib_dir(cfg), f"overlay_{cfg.mode}_{name}.json")
    write_overlay(out, view, size, name, stamp)
    print(f"[calib] {name}: overlay pos {view['pos']} look_at {view['look_at']} up {view['up']} "
          f"hfov {view['capture_fov']:.2f} deg -> {out}")
    vr = c.get("overlay_file")
    if vr:
        vr = os.path.normpath(os.path.join(os.path.dirname(cfg.path), vr))
        if os.path.exists(vr):
            write_overlay(vr, view, size, name, stamp)
            print(f"[calib] {name}: updated {vr}")
        else:
            print(f"[calib] {name}: overlay_file {vr} not found, copy {out} over by hand")
    if abs(r["K"][0, 2] - (size[0] - 1) / 2.0) > 2.0 or abs(r["K"][1, 2] - (size[1] - 1) / 2.0) > 2.0 \
            or np.max(np.abs(r["dist"])) > 1e-3:
        print(f"[calib] {name}: principal point / distortion are not what the overlay assumes -- the video must be "
              f"undistorted to the overlay pinhole")
    if cam.nominal is None or base is None:
        return
    ws = c.get("coverage", {})
    P = workspace_points(ws.get("workspace_min", [0.3, -0.4, 0.7]), ws.get("workspace_max", [0.9, 0.4, 1.1]))
    truth = project(cam.nominal, base.K, base.dist, P)
    ideal = project(cam.nominal, base.K, None, P)
    calib = project(r["T_world_cam"], r["K"], r["dist"], P)
    ov = project(r["T_world_cam"], overlay_K(view, size), None, P)
    for label, a, b in (("calibrated camera vs truth", calib, truth),
                        ("overlay vs undistorted video", ov, ideal),
                        ("overlay vs raw video (no undistortion)", ov, truth)):
        e = pixel_error(a, b, size)
        if e:
            print(f"[calib] {name}: {label}: {e[0]:.2f} px mean, {e[1]:.2f} px max over the workspace")


def calibrate(cfg, samples_path=None, role="perception"):
    """Moves the arm through calibration poses (or reuses saved samples), solves every camera and re-measures the reference tags."""
    cfg = _for_role(cfg, role)
    c = cfg.calibration
    target = make_target(cfg)
    ic = c.get("intrinsics", {})
    dist_terms = int(ic.get("dist_terms", 5))
    detector = TagDetector(cfg.board.dictionary, cfg.detection.get("detector", {}))
    cams = {cam.name: cam for cam in cfg.cameras}
    sources = {name: make_source(cam) for name, cam in cams.items()}
    base = {}
    for name, src in sources.items():
        cam = cams[name]
        if cam.source.get("type") == "realsense":
            w, h, K, dist = src.intrinsics()
            base[name] = (Intrinsics(width=w, height=h, K=K, dist=dist), "factory intrinsics")
            if cam.intrinsics_from_device:
                cam.intrinsics = base[name][0]
        elif cam.intrinsics_base is not None:
            base[name] = (cam.intrinsics_base, "intrinsics file")
    out_dir = calib_dir(cfg)
    os.makedirs(out_dir, exist_ok=True)
    if samples_path:
        samples = load_samples(samples_path, cams, target)
        print(f"[calib] re-solving from {samples_path}")
    else:
        samples = _collect(cfg, cams, sources, target)
        if samples is None:
            for s in sources.values():
                s.close()
            return
    stamp = time.strftime("%Y%m%d_%H%M%S")
    if not samples_path:
        tag = "" if role == "perception" else f"{role}_"
        save_samples(os.path.join(out_dir, f"samples_{cfg.mode}_{tag}{stamp}.npz"), samples, target)
    min_valid = int(c.get("capture", {}).get("min_valid_poses", 10))
    ref_points = [t.T[:3, 3] for t in cfg.reference_tags]
    for name, cam in cams.items():
        s = samples[name]
        if len(s) < min_valid:
            print(f"[calib] {name}: only {len(s)} valid poses (< {min_valid}), nothing written")
            continue
        intr = cam.intrinsics
        size = (intr.width, intr.height)
        est = _wants_intrinsics(cfg, target) and len(s) >= int(ic.get("min_views", 15))
        if _wants_intrinsics(cfg, target) and not est:
            print(f"[calib] {name}: only {len(s)} views, intrinsics kept ({cam.intrinsics_origin})")
        r = _solve_camera(s, target, intr, est, dist_terms, min_valid, name)
        per = r["rms_px"]
        print(f"[calib] {name}: {len(per)} poses, reprojection rms median {np.median(per):.3f} px, max {per.max():.3f} px")
        if est:
            b, label = base.get(name, (None, ""))
            _report_intrinsics(name, r, b, label, size, c.get("coverage", {}))
        if cam.nominal is not None and ref_points:
            print(f"[calib] {name}: vs nominal sim camera {workspace_drift(r['T_world_cam'], cam.nominal, ref_points) * 1000:.3f} mm"
                  f" (mean displacement at the reference tags)")
        with open(cam.extrinsics_path, "w") as f:
            yaml.safe_dump({"camera": name, "mode": cfg.mode, "date": stamp, "target": target.kind,
                            "n_poses": len(per), "rms_px_median": float(np.median(per)), "rms_px_max": float(per.max()),
                            "intrinsics": "estimated" if est else cam.intrinsics_origin,
                            "T_world_cam": to_list(r["T_world_cam"]), "T_ee_target": to_list(r["T_ee_target"])},
                           f, sort_keys=False)
        cam.T_world_cam = r["T_world_cam"]
        print(f"[calib] wrote {cam.extrinsics_path}")
        if est:
            path = cam.intrinsics_path or os.path.join(out_dir, f"intrinsics_{cfg.mode}_{name}.yaml")
            K, d = r["K"], r["dist"]
            with open(path, "w") as f:
                yaml.safe_dump({"camera": name, "date": stamp, "width": intr.width, "height": intr.height,
                                "fx": float(K[0, 0]), "fy": float(K[1, 1]), "cx": float(K[0, 2]), "cy": float(K[1, 2]),
                                "dist": [float(v) for v in d], "n_views": len(per),
                                "rms_px_median": float(np.median(per))}, f, sort_keys=False)
            cam.intrinsics = Intrinsics(width=intr.width, height=intr.height, K=K, dist=d)
            print(f"[calib] wrote {path}")
        if cam.role == "operator":
            _report_overlay(cfg, name, cam, r, size, stamp, base.get(name, (None, ""))[0])
    if cfg.reference_tags and cfg.reference.get("measured_path"):
        n_ref = int(c.get("capture", {}).get("reference_frames", 60))
        ref = measure_reference(sources, cams, detector, cfg.reference_tags, n_ref)
        if ref:
            _save_reference(cfg.reference["measured_path"], ref, stamp, cfg.mode)
            summary = ", ".join(f"{n}: {sorted(v)}" for n, v in ref.items())
            print(f"[calib] wrote {cfg.reference['measured_path']} (reference tags per camera -- {summary})")
        else:
            print("[calib] no reference tags seen, reference file not written")
    for s in sources.values():
        s.close()
