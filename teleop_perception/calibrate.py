import os
import time

import cv2
import numpy as np
import yaml
from scipy.optimize import least_squares

from . import wire
from .config import Intrinsics, calib_dir
from .detection import TagDetector
from .geometry import T_from_rvec, average_poses, inv_T, make_T, pose_error, rvec_from_T, to_list
from .reference import workspace_drift
from .sources import make_source


def tag_square(size):
    """Corners of a square tag in its own frame, OpenCV order."""
    h = size / 2.0
    return np.array([[-h, h, 0.0], [h, h, 0.0], [h, -h, 0.0], [-h, -h, 0.0]])


def generate_poses(T0, motion, rng):
    """Random poses around T0 inside the workspace box, ordered to keep travel short."""
    rng_pos = np.asarray(motion.get("pos_range_m", [0.08, 0.08, 0.06]), dtype=float)
    rot = np.radians(float(motion.get("rot_range_deg", 25.0)))
    lo = np.asarray(motion.get("workspace_min", [-np.inf] * 3), dtype=float)
    hi = np.asarray(motion.get("workspace_max", [np.inf] * 3), dtype=float)
    poses = []
    for _ in range(int(motion.get("n_poses", 18))):
        p = np.clip(T0[:3, 3] + rng.uniform(-rng_pos, rng_pos), lo, hi)
        axis = rng.normal(size=3)
        axis /= np.linalg.norm(axis)
        R, _ = cv2.Rodrigues(axis * rng.uniform(0.3, 1.0) * rot)
        poses.append(make_T(T0[:3, :3] @ R, p))
    ordered, cur = [], T0
    while poses:
        k = int(np.argmin([np.linalg.norm(P[:3, 3] - cur[:3, 3]) for P in poses]))
        cur = poses.pop(k)
        ordered.append(cur)
    return ordered


def _ippe(corners, size, K, dist):
    """Lower-error IPPE pose of a single square tag."""
    n, rvecs, tvecs, errs = cv2.solvePnPGeneric(tag_square(size), corners, K, dist, flags=cv2.SOLVEPNP_IPPE_SQUARE)
    k = int(np.argmin(np.ravel(errs)[:n]))
    return T_from_rvec(rvecs[k], tvecs[k])


def _hand_eye_init(T_we, T_ct):
    """Closed-form eye-to-hand start (Park & Martin on A Y = Y B): returns (T_world_cam, T_ee_tag)."""
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


def solve_hand_eye(samples, size, K, dist):
    """Camera-in-world and tag-on-gripper from [(T_world_ee, corners)], refined on reprojection error."""
    T_we = [T for T, _ in samples]
    T_ct = [_ippe(c, size, K, dist) for _, c in samples]
    T_wc0, T_et0 = _hand_eye_init(T_we, T_ct)
    T_cw0 = inv_T(T_wc0)
    obj = tag_square(size)

    def unpack(x):
        """Parameter vector to (T_cam_world, T_ee_tag)."""
        return T_from_rvec(x[0:3], x[3:6]), T_from_rvec(x[6:9], x[9:12])

    def residuals(x):
        """Tag corner reprojection error over all samples."""
        T_cw, T_et = unpack(x)
        res = []
        for T_we, c in samples:
            rv, tv = rvec_from_T(T_cw @ T_we @ T_et)
            proj, _ = cv2.projectPoints(obj, rv, tv, K, dist)
            res.append((proj.reshape(-1, 2) - c).ravel())
        return np.concatenate(res)

    r1, t1 = rvec_from_T(T_cw0)
    r2, t2 = rvec_from_T(T_et0)
    sol = least_squares(residuals, np.r_[r1.ravel(), t1.ravel(), r2.ravel(), t2.ravel()], loss="huber",
                        f_scale=1.0, x_scale="jac")
    T_cw, T_et = unpack(sol.x)
    per = np.sqrt(np.mean(sol.fun.reshape(len(samples), 4, 2) ** 2, axis=(1, 2)) * 2.0)
    return inv_T(T_cw), T_et, per


class ArmNotFollowing(RuntimeError):
    """The arm ignored the command stream, usually because the avatar dropped POLICY to HOLD."""


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


def face_cameras(arm, sources, cams, detector, tag_id, T0, motion, st):
    """Moves to the start position and turns the hand about its tool axis until the gripper tag faces the cameras."""
    T = T0.copy()
    if motion.get("start_pos") is not None:
        T[:3, 3] = np.asarray(motion["start_pos"], dtype=float)
    best, best_n = None, 0
    for yaw in motion.get("search_yaw_deg", [0, 45, -45, 90, -90, 135, -135, 180]):
        R, _ = cv2.Rodrigues(np.array([0.0, 0.0, np.radians(yaw)]))
        cand = make_T(T[:3, :3] @ R, T[:3, 3])
        if _settle(arm, cand, st) is None:
            print(f"[calib] start yaw {yaw:+d} deg: not reached, skipped")
            continue
        seen = sum(c is not None for c in _capture(sources, cams, detector, tag_id, 5).values())
        print(f"[calib] start yaw {yaw:+d} deg: tag seen by {seen}/{len(sources)} cameras")
        if seen > best_n:
            best, best_n = cand, seen
        if seen == len(sources):
            break
    if best is None:
        raise RuntimeError("gripper tag not visible from any start yaw -- check the cap and start_pos")
    return best


def _capture(sources, cams, detector, tag_id, n_frames, timeout=5.0):
    """Collects n_frames per camera and returns {camera: mean corners of tag_id or None}."""
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
            dets = detector.detect(f.gray)
            if tag_id in dets:
                got[name].append(dets[tag_id])
        time.sleep(0.002)
    return {c: (np.mean(v, axis=0) if len(v) >= max(1, n_frames // 2) else None) for c, v in got.items()}


def measure_reference(sources, cams, detector, ref_tags, n_frames):
    """Where each camera sees each reference tag right after calibration: median corners and depth per tag."""
    ids = {t.id: t for t in ref_tags}
    seen = {c: {} for c in sources}
    count = {c: 0 for c in sources}
    t_end = time.monotonic() + 10.0
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


def _collect(cfg, cams, sources, detector):
    """Engages the arm, finds a start yaw, visits the calibration poses and returns the samples (None if aborted)."""
    c = cfg.calibration
    gt = c["gripper_tag"]
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
        T0 = face_cameras(arm, sources, cams, detector, int(gt["id"]), T0, c.get("motion", {}), st)
        poses = generate_poses(T0, c.get("motion", {}), np.random.default_rng(int(c["motion"].get("seed", 0))))
        for k, P in enumerate(poses):
            T_meas = _settle(arm, P, st)
            if T_meas is None:
                print(f"[calib] pose {k + 1}/{len(poses)}: arm did not settle, skipped")
                continue
            t0 = time.monotonic()
            corners = _capture(sources, cams, detector, int(gt["id"]), int(cap.get("frames", 15)))
            T_mean = arm.measured_since(t0)
            T_mean = T_meas if T_mean is None else T_mean
            seen = [n for n, cr in corners.items() if cr is not None]
            for n in seen:
                samples[n].append((T_mean, corners[n]))
            print(f"[calib] pose {k + 1}/{len(poses)}: tag seen by {seen or 'no camera'}")
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


def load_samples(path, cams):
    """Samples saved by an earlier calibration run, {camera: [(T_world_ee, corners)]}."""
    z = np.load(path)
    return {n: list(zip(z[f"{n}_T"], z[f"{n}_corners"])) if f"{n}_T" in z else [] for n in cams}


def calibrate(cfg, samples_path=None):
    """Moves the arm through calibration poses (or reuses saved samples), solves every camera and re-measures the reference tags."""
    c = cfg.calibration
    gt = c["gripper_tag"]
    detector = TagDetector(cfg.board.dictionary, cfg.detection.get("detector", {}))
    cams = {cam.name: cam for cam in cfg.cameras}
    sources = {name: make_source(cam) for name, cam in cams.items()}
    for name, src in sources.items():
        if cams[name].intrinsics_from_device:
            w, h, K, dist = src.intrinsics()
            cams[name].intrinsics = Intrinsics(width=w, height=h, K=K, dist=dist)
    out_dir = calib_dir(cfg)
    os.makedirs(out_dir, exist_ok=True)
    if samples_path:
        samples = load_samples(samples_path, cams)
        print(f"[calib] re-solving from {samples_path}")
    else:
        samples = _collect(cfg, cams, sources, detector)
        if samples is None:
            for s in sources.values():
                s.close()
            return
    stamp = time.strftime("%Y%m%d_%H%M%S")
    if not samples_path:
        np.savez(os.path.join(out_dir, f"samples_{cfg.mode}_{stamp}.npz"),
                 **{f"{n}_T": np.array([s[0] for s in v]) for n, v in samples.items() if v},
                 **{f"{n}_corners": np.array([s[1] for s in v]) for n, v in samples.items() if v})
    min_valid = int(c.get("capture", {}).get("min_valid_poses", 10))
    ref_points = [t.T[:3, 3] for t in cfg.reference_tags]
    for name, cam in cams.items():
        s = samples[name]
        if len(s) < min_valid:
            print(f"[calib] {name}: only {len(s)} valid poses (< {min_valid}), extrinsics not written")
            continue
        intr = cam.intrinsics
        T_wc, T_et, per = solve_hand_eye(s, float(gt["size"]), intr.K, intr.dist)
        good = per <= max(1.0, 3.0 * float(np.median(per)))
        if not good.all() and good.sum() >= min_valid:
            print(f"[calib] {name}: dropping {int((~good).sum())} outlier poses")
            s = [x for x, g in zip(s, good) if g]
            T_wc, T_et, per = solve_hand_eye(s, float(gt["size"]), intr.K, intr.dist)
        print(f"[calib] {name}: {len(s)} poses, reprojection rms median {np.median(per):.3f} px, max {per.max():.3f} px")
        if cam.nominal is not None and ref_points:
            print(f"[calib] {name}: vs nominal sim camera {workspace_drift(T_wc, cam.nominal, ref_points) * 1000:.3f} mm"
                  f" (mean displacement at the reference tags)")
        with open(cam.extrinsics_path, "w") as f:
            yaml.safe_dump({"camera": name, "mode": cfg.mode, "date": stamp, "n_poses": len(s),
                            "rms_px_median": float(np.median(per)), "rms_px_max": float(per.max()),
                            "T_world_cam": to_list(T_wc), "T_ee_tag": to_list(T_et)}, f, sort_keys=False)
        cam.T_world_cam = T_wc
        print(f"[calib] wrote {cam.extrinsics_path}")
    if cfg.reference_tags and cfg.reference.get("measured_path"):
        n_ref = int(c.get("capture", {}).get("reference_frames", 60))
        ref = measure_reference(sources, cams, detector, cfg.reference_tags, n_ref)
        for s in sources.values():
            s.close()
        if ref:
            with open(cfg.reference["measured_path"], "w") as f:
                yaml.safe_dump({"date": stamp, "mode": cfg.mode, "cameras": ref}, f, sort_keys=False)
            summary = ", ".join(f"{n}: {sorted(v)}" for n, v in ref.items())
            print(f"[calib] wrote {cfg.reference['measured_path']} (reference tags per camera -- {summary})")
    else:
        for s in sources.values():
            s.close()
