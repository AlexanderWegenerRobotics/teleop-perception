import os
import re
import xml.etree.ElementTree as ET

import cv2
import numpy as np
import yaml

from .config import Tag
from .geometry import quat_from_R
from .targets import PX_PER_SQUARE, make_target

QUIET_CELLS = 1
PX_PER_CELL = 32
SLAB_DEPTH = 0.0005
V_TOP = 0.0


def marker_cells(dictionary):
    """Cells per marker side including the black border."""
    d = cv2.aruco.getPredefinedDictionary(getattr(cv2.aruco, dictionary))
    return d.markerSize + 2


def tag_image(dictionary, tag_id):
    """Marker image with a white quiet zone, as RGB."""
    d = cv2.aruco.getPredefinedDictionary(getattr(cv2.aruco, dictionary))
    cells = d.markerSize + 2
    img = cv2.aruco.generateImageMarker(d, int(tag_id), cells * PX_PER_CELL)
    pad = QUIET_CELLS * PX_PER_CELL
    img = cv2.copyMakeBorder(img, pad, pad, pad, pad, cv2.BORDER_CONSTANT, value=255)
    return cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)


def _slab(hw, hh):
    """Vertices, texcoords and outward faces of a thin rectangular slab whose top face is z = 0."""
    top = [(-hw, hh), (hw, hh), (hw, -hh), (-hw, -hh)]
    verts = [(x, y, 0.0) for x, y in top] + [(x, y, -SLAB_DEPTH) for x, y in top]
    uv_top = [(0.0, V_TOP), (1.0, V_TOP), (1.0, 1.0 - V_TOP), (0.0, 1.0 - V_TOP)]
    uvs = uv_top + uv_top
    quads = [(0, 1, 2, 3), (7, 6, 5, 4), (0, 4, 5, 1), (1, 5, 6, 2), (2, 6, 7, 3), (3, 7, 4, 0)]
    V = np.array(verts)
    c = V.mean(axis=0)
    faces = []
    for a, b, cc, d in quads:
        for tri in ((a, b, cc), (a, cc, d)):
            p0, p1, p2 = V[list(tri)]
            n = np.cross(p1 - p0, p2 - p0)
            if n @ ((p0 + p1 + p2) / 3.0 - c) < 0:
                tri = (tri[0], tri[2], tri[1])
            faces.append(tri)
    return verts, uvs, faces


def _fmt(vals):
    """Space-separated numbers for MJCF attributes."""
    return " ".join(f"{v:.6g}" for v in vals)


def add_slab(asset, body, name, png, hw, hh, T):
    """Adds texture, material, mesh and a visual-only geom showing png on a slab at pose T."""
    verts, uvs, faces = _slab(hw, hh)
    ET.SubElement(asset, "texture", name=f"{name}_tex", type="2d", file=png)
    ET.SubElement(asset, "material", name=f"{name}_mat", texture=f"{name}_tex", rgba="1 1 1 1", specular="0",
                  shininess="0", reflectance="0")
    ET.SubElement(asset, "mesh", name=f"{name}_mesh", vertex=_fmt(np.ravel(verts)), texcoord=_fmt(np.ravel(uvs)),
                  face=" ".join(str(i) for i in np.ravel(faces)))
    ET.SubElement(body, "geom", name=name, type="mesh", mesh=f"{name}_mesh", material=f"{name}_mat",
                  pos=_fmt(T[:3, 3]), quat=_fmt(quat_from_R(T[:3, :3])),
                  contype="0", conaffinity="0", group="2", density="0")


def add_tag(asset, body, tag, dictionary, prefix, tex_dir):
    """Adds one tag with its quiet zone as a textured slab."""
    cells = marker_cells(dictionary)
    half = tag.size / 2.0 * (cells + 2 * QUIET_CELLS) / cells
    png = f"tag_{tag.id}.png"
    path = os.path.join(tex_dir, png)
    if not os.path.exists(path):
        cv2.imwrite(path, cv2.cvtColor(tag_image(dictionary, tag.id), cv2.COLOR_RGB2BGR))
    add_slab(asset, body, f"{prefix}tag_{tag.id}", png, half, half, tag.T)


def add_charuco(asset, body, target, T, prefix, tex_dir):
    """Adds the ChArUco board with its margin as a textured slab."""
    png = "charuco_board.png"
    img = target.image()
    cv2.imwrite(os.path.join(tex_dir, png), img)
    m_per_px = target.square / PX_PER_SQUARE
    add_slab(asset, body, f"{prefix}charuco", png, img.shape[1] * m_per_px / 2.0, img.shape[0] * m_per_px / 2.0, T)


def write_print(target, path, dpi=600):
    """Writes the target at true size as a PDF for printing at 100 %."""
    from PIL import Image
    px = int(round(target.square / 0.0254 * dpi))
    img = target.image(px)
    Image.fromarray(img).save(path, resolution=float(dpi))
    return path


def _find_body(root, name):
    """First <body> with the given name."""
    for b in root.iter("body"):
        if b.get("name") == name:
            return b
    raise KeyError(f"body '{name}' not found")


def tag_model(src, dst, body_name, tags, dictionary, prefix, mount_stl=None, extra=None):
    """Copies an MJCF model and puts tags, an optional printed mount and optional extra geoms on one body."""
    with open(src, "r", encoding="utf-8") as f:
        text = re.sub(r"<!--.*?-->", "", f.read(), flags=re.DOTALL)
    root = ET.fromstring(text)
    tree = ET.ElementTree(root)
    src_dir = os.path.dirname(os.path.abspath(src))
    dst_dir = os.path.dirname(os.path.abspath(dst))
    tex_dir = os.path.join(dst_dir, "tags")
    os.makedirs(tex_dir, exist_ok=True)
    comp = root.find("compiler")
    if comp is None:
        comp = ET.Element("compiler")
        root.insert(0, comp)
    meshdir = os.path.normpath(os.path.join(src_dir, comp.get("meshdir", "")))
    old_texdir = os.path.normpath(os.path.join(src_dir, comp.get("texturedir", "")))
    comp.set("meshdir", os.path.relpath(meshdir, dst_dir).replace(os.sep, "/"))
    comp.set("texturedir", "tags")
    asset = root.find("asset")
    if asset is None:
        asset = ET.Element("asset")
        root.insert(1, asset)
    for t in asset.findall("texture"):
        if t.get("file"):
            t.set("file", os.path.relpath(os.path.join(old_texdir, t.get("file")), tex_dir).replace(os.sep, "/"))
    body = _find_body(root, body_name)
    for tag in tags:
        add_tag(asset, body, tag, dictionary, prefix, tex_dir)
    if extra is not None:
        extra(asset, body, tex_dir)
    if mount_stl:
        rel = os.path.relpath(os.path.abspath(mount_stl), meshdir).replace(os.sep, "/")
        ET.SubElement(asset, "material", name=f"{prefix}mount_mat", rgba="0.15 0.15 0.15 1", specular="0.1")
        ET.SubElement(asset, "mesh", name=f"{prefix}mount", file=rel)
        ET.SubElement(body, "geom", name=f"{prefix}mount", type="mesh", mesh=f"{prefix}mount",
                      material=f"{prefix}mount_mat", contype="0", conaffinity="0", group="2", density="0")
    ET.indent(tree, space="  ")
    tree.write(dst, encoding="unicode")
    return dst


def reference_model(dst, ref_tags, dictionary):
    """Static model holding the reference tags at their world poses."""
    root = ET.Element("mujoco", model="perception_reference")
    ET.SubElement(root, "compiler", angle="radian", texturedir="tags")
    asset = ET.SubElement(root, "asset")
    wb = ET.SubElement(root, "worldbody")
    body = ET.SubElement(wb, "body", name="perception_reference", pos="0 0 0")
    ET.SubElement(body, "inertial", pos="0 0 0", mass="0.001", diaginertia="1e-6 1e-6 1e-6")
    tex_dir = os.path.join(os.path.dirname(os.path.abspath(dst)), "tags")
    os.makedirs(tex_dir, exist_ok=True)
    for t in ref_tags:
        add_tag(asset, body, t, dictionary, "ref_", tex_dir)
    tree = ET.ElementTree(root)
    ET.indent(tree, space="  ")
    tree.write(dst, encoding="unicode")
    return dst


def generate(cfg):
    """Writes tagged object models, the reference model, the tagged gripper and a tagged task config."""
    sim = cfg.sim
    root = sim["root"]
    run_dir = os.path.join(root, sim.get("run_dir", "build"))
    out_dir = os.path.join(root, sim["generated_dir"])
    os.makedirs(out_dir, exist_ok=True)
    task_path = os.path.join(root, sim["task_config"])
    with open(task_path) as f:
        task = yaml.safe_load(f)
    by_name = {o["name"]: o for o in task.get("objects", [])}
    written = []
    for obj in cfg.board.objects:
        if obj.name not in by_name:
            raise KeyError(f"object '{obj.name}' is not in {task_path}")
        entry = by_name[obj.name]
        src = os.path.normpath(os.path.join(run_dir, entry["model_path"]))
        dst = os.path.join(out_dir, f"{obj.name}_tagged.xml")
        tag_model(src, dst, entry["body_name"], obj.tags.values(), cfg.board.dictionary, f"{obj.name}_")
        entry["model_path"] = os.path.relpath(dst, run_dir).replace(os.sep, "/")
        written.append(dst)
    tagged_task = os.path.join(root, sim["tagged_task_config"])
    _dump(task, tagged_task)
    written.append(tagged_task)
    ref_path = reference_model(os.path.join(out_dir, "perception_reference.xml"), cfg.reference_tags,
                               cfg.board.dictionary)
    written.append(ref_path)
    hand_path = None
    if cfg.calibration and sim.get("gripper_model"):
        hand_path = _hand(cfg, os.path.join(root, sim["gripper_model"]), out_dir, written)
    written += sim_configs(cfg, run_dir, tagged_task, ref_path, hand_path)
    return written


def _hand(cfg, hand_src, out_dir, written):
    """Writes the hand with the active calibration target on its printed mount, plus the print files."""
    from .mount import write_mount
    target = make_target(cfg)
    body = cfg.calibration.get("mount", {}).get("body", "hand")
    print_dir = os.path.join(os.path.dirname(os.path.dirname(cfg.path)), "hardware")
    mount_stl, print_stl, T, _ = write_mount(cfg, target, hand_src, out_dir, print_dir)
    written += [mount_stl, print_stl]
    T[:3, 3] += cfg.board.standoff * T[:3, 2]
    dst = os.path.join(out_dir, "hand_tagged.xml")
    if target.kind == "charuco":
        written.append(write_print(target, os.path.join(print_dir, "charuco_board_print.pdf")))
        extra = lambda asset, b, tex_dir: add_charuco(asset, b, target, T, "calib_", tex_dir)
        path = tag_model(hand_src, dst, body, [], cfg.board.dictionary, "calib_", mount_stl, extra)
    else:
        tag = Tag(id=target.id, size=target.size, T=T)
        path = tag_model(hand_src, dst, body, [tag], cfg.board.dictionary, "calib_", mount_stl)
    written.append(path)
    return path


class _Dumper(yaml.SafeDumper):
    """YAML dumper that writes short numeric lists inline, like the hand-written sim configs."""


def _list_repr(dumper, data):
    """Inline style for lists of plain numbers."""
    flow = all(isinstance(v, (int, float)) and not isinstance(v, bool) for v in data)
    return dumper.represent_sequence("tag:yaml.org,2002:seq", data, flow_style=flow)


_Dumper.add_representer(list, _list_repr)


def _dump(data, path):
    """Writes a config file in block style with inline numeric lists."""
    with open(path, "w") as f:
        yaml.dump(data, f, Dumper=_Dumper, sort_keys=False, default_flow_style=False)


def _perception_cameras(cfg):
    """Sim camera entries and shared-memory stream entries for the perception cameras."""
    cams, streams = [], []
    for c in cfg.cameras:
        if c.nominal is None or c.source.get("type") != "shm":
            continue
        gt = c.intrinsics_base or c.intrinsics
        fovy = float(np.degrees(2.0 * np.arctan((gt.height / 2.0) / gt.K[1, 1])))
        cams.append({"name": c.name, "enabled": True, "type": "fixed",
                     "pos": [float(v) for v in c.nominal[:3, 3]],
                     "look_at": [float(v) for v in c.nominal_look_at], "fovy": round(fovy, 4)})
        streams.append({"camera": c.name, "shm_name": c.source["name"], "eye": "aux",
                        "width": int(c.intrinsics.width), "height": int(c.intrinsics.height)})
    return cams, streams


def sim_configs(cfg, run_dir, tagged_task, ref_path, hand_path):
    """Writes perception variants of the simulator's sim, pipeline and robot configs."""
    sim = cfg.sim
    root = sim["root"]
    rel = lambda p: os.path.relpath(p, run_dir).replace(os.sep, "/")
    cams, streams = _perception_cameras(cfg)
    names = {c["name"] for c in cams}
    out = []
    with open(os.path.join(root, sim["base_sim_config"])) as f:
        sc = yaml.safe_load(f)
    sc["simulation"]["task_config"] = rel(tagged_task)
    if hand_path:
        for d in sc.get("devices", []):
            if d.get("name") == sim.get("gripper_device", "hand_left"):
                d["model_path"] = rel(hand_path)
    sc["objects"] = [o for o in sc.get("objects", []) if o.get("name") != "perception_reference"]
    sc["objects"].append({"name": "perception_reference", "type": "static", "model_path": rel(ref_path),
                          "pose": {"position": [0, 0, 0], "orientation": [1, 0, 0, 0]}})
    sc["cameras"] = [c for c in sc.get("cameras", []) if c.get("name") not in names] + cams
    path = os.path.join(root, sim["out_sim_config"])
    _dump(sc, path)
    out.append(path)
    with open(os.path.join(root, sim["base_pipeline_config"])) as f:
        pc = yaml.safe_load(f)
    pc["stream_cameras"] = [s for s in pc.get("stream_cameras", []) if s.get("camera") not in names] + streams
    path = os.path.join(root, sim["out_pipeline_config"])
    _dump(pc, path)
    out.append(path)
    if sim.get("base_robot_config") and sim.get("out_robot_config"):
        with open(os.path.join(root, sim["base_robot_config"])) as f:
            rc = yaml.safe_load(f)
        gt = cfg.network.get("ground_truth", {})
        rc.setdefault("avatar", {})["scene_objects"] = {"enabled": True, "host": "127.0.0.1",
                                                        "port": int(gt.get("port", 7200))}
        if sim.get("authority_stale_ms"):
            for d in rc.get("devices", []):
                if d.get("type") == "arm":
                    d.setdefault("control", {})["authority_stale_ms"] = float(sim["authority_stale_ms"])
        path = os.path.join(root, sim["out_robot_config"])
        _dump(rc, path)
        out.append(path)
    return out
