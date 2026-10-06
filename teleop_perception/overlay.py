import json
import os
import re

import cv2
import numpy as np


def overlay_view(T_world_cam, K, size):
    """Static viewpoint (pos, look_at, up) and horizontal FOV as the VR ghost overlay uses them."""
    pos = T_world_cam[:3, 3]
    hfov = float(np.degrees(2.0 * np.arctan(size[0] / (2.0 * K[0, 0]))))
    return {"pos": [round(float(v), 5) for v in pos],
            "look_at": [round(float(v), 5) for v in pos + T_world_cam[:3, 2]],
            "up": [round(float(v), 5) for v in -T_world_cam[:3, 1]],
            "capture_fov": round(hfov, 4)}


def overlay_K(view, size):
    """Pinhole matrix the overlay renders with: square pixels, centred principal point."""
    w, h = size
    f = (w / 2.0) / np.tan(np.radians(view["capture_fov"]) / 2.0)
    return np.array([[f, 0.0, (w - 1) / 2.0], [0.0, f, (h - 1) / 2.0], [0.0, 0.0, 1.0]])


def project(T_world_cam, K, dist, P):
    """Pixels of world points P seen by a camera."""
    T = np.linalg.inv(T_world_cam)
    rv, _ = cv2.Rodrigues(T[:3, :3])
    uv, _ = cv2.projectPoints(np.asarray(P, dtype=float), rv, T[:3, 3], K, np.zeros(5) if dist is None else dist)
    return uv.reshape(-1, 2)


def workspace_points(lo, hi, n=7):
    """Grid of points in the workspace box."""
    axes = [np.linspace(a, b, n) for a, b in zip(lo, hi)]
    return np.stack(np.meshgrid(*axes, indexing="ij"), axis=-1).reshape(-1, 3)


def pixel_error(a, b, size):
    """Mean and max distance between two pixel sets, over points inside the image."""
    w, h = size
    m = (b[:, 0] >= 0) & (b[:, 0] < w) & (b[:, 1] >= 0) & (b[:, 1] < h)
    if not m.any():
        return None
    e = np.linalg.norm(a[m] - b[m], axis=1)
    return float(e.mean()), float(e.max())


def write_overlay(path, view, size, camera, stamp):
    """Writes the overlay viewpoint for one camera; updates an existing VR overlay config in place."""
    data = {}
    if os.path.exists(path):
        with open(path) as f:
            data = json.load(f)
    vp = data.setdefault("viewpoint", {})
    vp["mode"] = "static"
    vp["static"] = {"pos": view["pos"], "look_at": view["look_at"], "up": view["up"]}
    data["_viewpoint_comment"] = (f"STATIC viewpoint written by teleop-perception 'calibrate --role operator' for "
                                  f"camera '{camera}' ({stamp}). Recalibrate instead of editing by hand.")
    data["_capture_fov_comment"] = ("HORIZONTAL FOV from the calibrated focal length. The overlay assumes square "
                                    "pixels, a centred principal point and no lens distortion, so the video must "
                                    "be undistorted to the same pinhole.")
    data["capture_fov"] = view["capture_fov"]
    data["stereo_capture_fov"] = view["capture_fov"]
    data["render_target_width"], data["render_target_height"] = int(size[0]), int(size[1])
    text = re.sub(r"\[\s*([-0-9.eE,\s]+?)\s*\]", lambda m: "[" + ", ".join(v.strip() for v in m.group(1).split(",")) + "]",
                  json.dumps(data, indent=4))
    with open(path, "w") as f:
        f.write(text + "\n")
    return path
