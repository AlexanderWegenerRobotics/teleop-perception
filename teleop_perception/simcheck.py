import os

import cv2
import numpy as np
import yaml

from .geometry import T_from_pos_quat, mj_quat_look_at, pose_error
from .pipeline import Pipeline
from .reference import reference_reprojection


def _yaw_quat(yaw):
    """Quaternion (w, x, y, z) of a rotation about world z."""
    return [np.cos(yaw / 2.0), 0.0, 0.0, np.sin(yaw / 2.0)]


def build_scene(cfg, table_z=0.7):
    """MuJoCo model with the tagged objects, the reference tags, a table and the perception cameras."""
    import mujoco
    sim = cfg.sim
    root = sim["root"]
    out_dir = os.path.join(root, sim["generated_dir"])
    with open(os.path.join(root, sim["task_config"])) as f:
        task = yaml.safe_load(f)
    by_name = {o["name"]: o for o in task["objects"]}
    spec = mujoco.MjSpec()
    spec.visual.global_.offwidth = max(c.intrinsics.width for c in cfg.cameras)
    spec.visual.global_.offheight = max(c.intrinsics.height for c in cfg.cameras)
    spec.visual.quality.offsamples = 4
    spec.visual.headlight.ambient = [0.4, 0.4, 0.4]
    spec.visual.headlight.diffuse = [0.5, 0.5, 0.5]
    wb = spec.worldbody
    wb.add_light(pos=[0.7, 0.0, 2.0], dir=[0.0, 0.0, -1.0], diffuse=[0.6, 0.6, 0.6])
    wb.add_geom(type=mujoco.mjtGeom.mjGEOM_BOX, size=[0.6, 0.6, 0.02], pos=[0.7, 0.0, table_z - 0.02],
                rgba=[0.55, 0.55, 0.58, 1.0])
    for c in cfg.cameras:
        if c.nominal is None:
            continue
        gt = c.intrinsics_base or c.intrinsics
        fovy = np.degrees(2.0 * np.arctan((gt.height / 2.0) / gt.K[1, 1]))
        pos = c.nominal[:3, 3]
        wb.add_camera(name=c.name, pos=pos, quat=mj_quat_look_at(pos, pos + c.nominal[:3, 2]), fovy=fovy)
    bodies = {}
    for obj in cfg.board.objects:
        entry = by_name[obj.name]
        child = mujoco.MjSpec.from_file(os.path.join(out_dir, f"{obj.name}_tagged.xml"))
        frame = wb.add_frame(pos=entry["pose"]["position"], quat=entry["pose"]["orientation"])
        frame.attach_body(child.body(entry["body_name"]), f"{obj.name}_", "")
        bodies[obj.name] = f"{obj.name}_{entry['body_name']}"
    ref = mujoco.MjSpec.from_file(os.path.join(out_dir, "perception_reference.xml"))
    wb.add_frame().attach_body(ref.body("perception_reference"), "ref_", "")
    model = spec.compile()
    return model, bodies, by_name


def _gt(model, data, bodies):
    """World pose of every object body."""
    import mujoco
    out = {}
    for name, b in bodies.items():
        i = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, b)
        out[name] = T_from_pos_quat(data.xpos[i], data.xquat[i])
    return out


def _place(model, data, bodies, by_name, rng, jitter):
    """Spawns each object at its task pose plus random xy offset and yaw."""
    import mujoco
    for name, b in bodies.items():
        bid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, b)
        if model.body_jntadr[bid] < 0:
            continue
        adr = model.jnt_qposadr[model.body_jntadr[bid]]
        p = np.array(by_name[name]["pose"]["position"], dtype=float)
        q = np.array(by_name[name]["pose"]["orientation"], dtype=float)
        if jitter and name != "board":
            p[:2] += rng.uniform(-jitter, jitter, 2)
            q = np.array(_yaw_quat(rng.uniform(0, 2 * np.pi)))
        data.qpos[adr:adr + 3] = p
        data.qpos[adr + 3:adr + 7] = q
    mujoco.mj_forward(model, data)


def _shift_tags(model, rng, sigma_m, sigma_deg=1.0):
    """Shifts and turns every object tag within its face, like a hand-placed sticker."""
    import mujoco
    for g in range(model.ngeom):
        n = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, g) or ""
        if "tag_" not in n or n.startswith("ref_") or n.startswith("calib_"):
            continue
        m = model.geom_dataid[g]
        verts = model.mesh_vert[model.mesh_vertadr[m]:model.mesh_vertadr[m] + model.mesh_vertnum[m]]
        k = int(np.argmin(np.ptp(verts, axis=0)))
        u, v = [i for i in range(3) if i != k]
        R = np.zeros(9)
        mujoco.mju_quat2Mat(R, model.geom_quat[g])
        R = R.reshape(3, 3)
        dx, dy = rng.normal(0.0, sigma_m, 2)
        model.geom_pos[g] += R[:, u] * dx + R[:, v] * dy
        a = np.radians(rng.normal(0.0, sigma_deg))
        q = model.geom_quat[g].copy()
        dq = np.zeros(4)
        dq[0] = np.cos(a / 2)
        dq[1 + k] = np.sin(a / 2)
        out = np.zeros(4)
        mujoco.mju_mulQuat(out, q, dq)
        model.geom_quat[g] = out


def _degrade(img, rng, noise_px, blur_px):
    """Adds blur and pixel noise to a rendered image."""
    if blur_px > 0:
        img = cv2.GaussianBlur(img, (0, 0), blur_px)
    if noise_px > 0:
        img = np.clip(img.astype(np.float32) + rng.normal(0.0, noise_px, img.shape), 0, 255).astype(np.uint8)
    return img


def run(cfg, scenes=5, jitter=0.05, seed=0, tag_shift_mm=0.0, noise=0.0, blur=0.0, save_dir=None):
    """Renders random scenes, runs the pipeline and prints the error against MuJoCo ground truth."""
    import mujoco
    rng = np.random.default_rng(seed)
    model, bodies, by_name = build_scene(cfg)
    if tag_shift_mm > 0:
        _shift_tags(model, rng, tag_shift_mm / 1000.0)
    data = mujoco.MjData(model)
    cams = {c.name: c for c in cfg.cameras if c.nominal is not None}
    for c in cams.values():
        c.T_world_cam, c.extrinsics_origin = c.nominal, "nominal"
    renderer = {c.name: mujoco.Renderer(model, c.intrinsics.height, c.intrinsics.width) for c in cams.values()}
    rows, cam_drift = [], []
    pipe = Pipeline(cfg)
    pipe.filter = None
    for s in range(scenes):
        _place(model, data, bodies, by_name, rng, jitter if s > 0 else 0.0)
        gt = _gt(model, data, bodies)
        frames = {}
        for name, r in renderer.items():
            r.update_scene(data, camera=name)
            rgb = _degrade(r.render(), rng, noise, blur)
            frames[name] = (cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY), s)
            if save_dir:
                os.makedirs(save_dir, exist_ok=True)
                cv2.imwrite(os.path.join(save_dir, f"scene{s:02d}_{name}.png"), cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR))
        fused, results = pipe.step(frames, cams)
        for r in results:
            intr = cams[r.camera].intrinsics
            ref = reference_reprojection(cfg.reference_tags, r.detections, cams[r.camera].T_world_cam, intr.K, intr.dist)
            if ref is not None:
                cam_drift.append((r.camera, ref[1], ref[0], ref[2]))
            for e in r.estimates:
                T = cams[r.camera].T_world_cam @ e.T_cam_obj
                dp, dr = pose_error(T, gt[e.name])
                rows.append((s, r.camera, e.name, len(e.tag_ids), e.rms_px, dp * 1000.0, dr, e.ambiguous))
        for w in fused:
            dp, dr = pose_error(w.T_world_obj, gt[w.name])
            rows.append((s, "fused", w.name, w.n_tags, w.rms_px, dp * 1000.0, dr, w.ambiguous))
    for r in renderer.values():
        r.close()
    report(rows, cam_drift, cfg, scenes)
    return rows


def report(rows, cam_drift, cfg, scenes):
    """Prints detection rate and error statistics per object and source."""
    names = [o.name for o in cfg.board.objects]
    sources = sorted({r[1] for r in rows}, key=lambda x: (x == "fused", x))
    print(f"\n{'object':10s} {'source':18s} {'found':>6s} {'tags':>5s} {'rms px':>7s} "
          f"{'pos med':>8s} {'pos p95':>8s} {'pos max':>8s} {'rot med':>8s} {'rot max':>8s}")
    for n in names:
        for src in sources:
            rs = [r for r in rows if r[2] == n and r[1] == src]
            if not rs:
                print(f"{n:10s} {src:18s} {0:>3d}/{scenes:<2d}")
                continue
            p = np.array([r[5] for r in rs])
            a = np.array([r[6] for r in rs])
            print(f"{n:10s} {src:18s} {len(rs):>3d}/{scenes:<2d} {np.mean([r[3] for r in rs]):5.1f} "
                  f"{np.mean([r[4] for r in rs]):7.3f} {np.median(p):7.2f}mm {np.percentile(p, 95):7.2f}mm "
                  f"{p.max():7.2f}mm {np.median(a):7.2f}d {a.max():7.2f}d")
    if cam_drift:
        print("\nreference tags under the camera's extrinsics (what the drift check sees):")
        for cam in sorted({c[0] for c in cam_drift}):
            d = [c for c in cam_drift if c[0] == cam]
            print(f"  {cam}: {np.mean([c[3] for c in d]):.1f} tags visible, "
                  f"{np.median([c[2] for c in d]):.3f} px, {np.median([c[1] for c in d]):.3f} mm")
