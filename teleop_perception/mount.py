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


def build_mount(mesh_paths, m, plate_size, plate_thickness, offset_z=0.0):
    """Slide-on cap for the hand's +y end with a flat plate; returns (mesh in hand frame, T_hand_target)."""
    pts = _load_points(mesh_paths)
    y0 = float(m.get("grip_from", 0.075))
    c = float(m.get("clearance", 0.0003))
    wall = float(m.get("wall", 0.0025))
    w, h = plate_size
    end = pts[pts[:, 1] >= y0 - 0.02]
    inner = _dilated_hull(end, c)
    outer = _dilated_hull(end[end[:, 1] >= y0], c + wall)
    y_face = float(end[:, 1].max()) + c + wall
    z_c = 0.5 * (float(end[:, 2].min()) + float(end[:, 2].max())) + offset_z
    pad = trimesh.creation.box(extents=[w, plate_thickness + wall, h],
                               transform=make_T(np.eye(3), [0.0, y_face - (plate_thickness + wall) / 2.0, z_c]))
    keep = trimesh.creation.box(extents=[1.0, y_face - y0, 1.0],
                                transform=make_T(np.eye(3), [0.0, (y_face + y0) / 2.0, 0.0]))
    solid = trimesh.boolean.union([outer, pad], engine="manifold")
    cap = trimesh.boolean.difference([solid, inner], engine="manifold")
    cap = trimesh.boolean.intersection([cap, keep], engine="manifold")
    R_target = np.column_stack([[-1.0, 0.0, 0.0], [0.0, 0.0, 1.0], [0.0, 1.0, 0.0]])
    return cap, make_T(R_target, [0.0, y_face, z_c])


def print_frame(mesh_hand, T_hand_target):
    """Mount turned so the target face lies on the print bed."""
    T = np.linalg.inv(T_hand_target)
    flip = make_T(np.diag([1.0, -1.0, -1.0]), [0.0, 0.0, 0.0])
    out = mesh_hand.copy()
    out.apply_transform(flip @ T)
    out.apply_translation([0.0, 0.0, -out.bounds[0][2]])
    return out


def plate_spec(cfg, target):
    """Plate size, thickness and offset for the active calibration target."""
    c = cfg.calibration
    if target.kind == "charuco":
        p = c["charuco"]
        return target.extent, float(p.get("plate_thickness", 0.004)), float(p.get("plate_offset_z", 0.0))
    p = c["gripper_tag"]
    plate = float(p.get("plate", 0.056))
    if plate < target.size * 8.0 / 6.0:
        raise ValueError(f"plate {plate * 1000:.1f} mm is smaller than the tag with its quiet zone")
    return (plate, plate), float(p.get("plate_thickness", 0.003)), 0.0


def write_mount(cfg, target, hand_model_path, out_dir, print_dir):
    """Builds the mount for the active target from the simulator's hand meshes, writes sim and print STLs."""
    import xml.etree.ElementTree as ET
    import re
    m = cfg.calibration["mount"]
    with open(hand_model_path, "r", encoding="utf-8") as f:
        root = ET.fromstring(re.sub(r"<!--.*?-->", "", f.read(), flags=re.DOTALL))
    comp = root.find("compiler")
    meshdir = os.path.join(os.path.dirname(os.path.abspath(hand_model_path)),
                           comp.get("meshdir", "") if comp is not None else "")
    paths = [os.path.join(meshdir, p) for p in m["meshes"]]
    size, thickness, offset = plate_spec(cfg, target)
    cap, T_target = build_mount(paths, m, size, thickness, offset)
    os.makedirs(out_dir, exist_ok=True)
    os.makedirs(print_dir, exist_ok=True)
    sim_stl = os.path.join(out_dir, f"{target.kind}_mount.stl")
    cap.export(sim_stl)
    print_stl = os.path.join(print_dir, f"{target.kind}_mount_print.stl")
    print_frame(cap, T_target).export(print_stl)
    return sim_stl, print_stl, T_target, cap
