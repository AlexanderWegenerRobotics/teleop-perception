import os

import numpy as np
import trimesh

from .geometry import make_T


def _load_points(paths):
    """Vertices of the hand housing meshes, in the hand body frame."""
    return np.vstack([trimesh.load(p, force="mesh").vertices for p in paths])


def _dilated_hull(points, r):
    """Convex hull grown outward by about r."""
    dirs = trimesh.creation.icosphere(subdivisions=1).vertices
    grown = (points[:, None, :] + r * dirs[None, :, :]).reshape(-1, 3)
    return trimesh.Trimesh(grown).convex_hull


def build_mount(mesh_paths, m, tag_size):
    """Slide-on cap for the hand's +y end with a flat tag plate; returns (mesh in hand frame, T_hand_tag)."""
    pts = _load_points(mesh_paths)
    y0 = float(m.get("grip_from", 0.075))
    c = float(m.get("clearance", 0.0003))
    wall = float(m.get("wall", 0.0025))
    plate = float(m.get("plate", 0.056))
    t_plate = float(m.get("plate_thickness", 0.003))
    end = pts[pts[:, 1] >= y0 - 0.02]
    inner = _dilated_hull(end, c)
    outer = _dilated_hull(end[end[:, 1] >= y0], c + wall)
    y_face = float(end[:, 1].max()) + c + wall
    z_mid = 0.5 * (float(end[:, 2].min()) + float(end[:, 2].max()))
    pad = trimesh.creation.box(extents=[plate, t_plate + wall, plate],
                               transform=make_T(np.eye(3), [0.0, y_face - (t_plate + wall) / 2.0, z_mid]))
    keep = trimesh.creation.box(extents=[1.0, y_face - y0, 1.0],
                                transform=make_T(np.eye(3), [0.0, (y_face + y0) / 2.0, 0.0]))
    solid = trimesh.boolean.union([outer, pad], engine="manifold")
    cap = trimesh.boolean.difference([solid, inner], engine="manifold")
    cap = trimesh.boolean.intersection([cap, keep], engine="manifold")
    R_tag = np.column_stack([[-1.0, 0.0, 0.0], [0.0, 0.0, 1.0], [0.0, 1.0, 0.0]])
    if plate < tag_size * 8.0 / 6.0:
        raise ValueError(f"plate {plate * 1000:.1f} mm is smaller than the tag with its quiet zone")
    return cap, make_T(R_tag, [0.0, y_face, z_mid])


def print_frame(mesh_hand, T_hand_tag):
    """Mount turned so the tag face lies on the print bed."""
    T = np.linalg.inv(T_hand_tag)
    flip = make_T(np.diag([1.0, -1.0, -1.0]), [0.0, 0.0, 0.0])
    out = mesh_hand.copy()
    out.apply_transform(flip @ T)
    out.apply_translation([0.0, 0.0, -out.bounds[0][2]])
    return out


def write_mount(cfg, hand_model_path, out_dir, print_dir):
    """Builds the mount from the simulator's hand meshes, writes the sim and print STLs, returns the tag pose."""
    import xml.etree.ElementTree as ET
    import re
    gt = cfg.calibration["gripper_tag"]
    m = gt["mount"]
    with open(hand_model_path, "r", encoding="utf-8") as f:
        root = ET.fromstring(re.sub(r"<!--.*?-->", "", f.read(), flags=re.DOTALL))
    comp = root.find("compiler")
    meshdir = os.path.join(os.path.dirname(os.path.abspath(hand_model_path)),
                           comp.get("meshdir", "") if comp is not None else "")
    paths = [os.path.join(meshdir, p) for p in m["meshes"]]
    cap, T_tag = build_mount(paths, m, float(gt["size"]))
    os.makedirs(out_dir, exist_ok=True)
    os.makedirs(print_dir, exist_ok=True)
    sim_stl = os.path.join(out_dir, "gripper_tag_mount.stl")
    cap.export(sim_stl)
    print_stl = os.path.join(print_dir, "gripper_tag_mount_print.stl")
    print_frame(cap, T_tag).export(print_stl)
    return sim_stl, print_stl, T_tag, cap
