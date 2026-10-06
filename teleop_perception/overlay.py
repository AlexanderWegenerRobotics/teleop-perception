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


def pinhole_out(K, dist, size, samples=64):
    """Square-pixel, centred pinhole for the undistorted video, zoomed just enough to leave no empty border."""
    w, h = size
    u = np.r_[np.linspace(0, w - 1, samples), np.full(samples, w - 1), np.linspace(0, w - 1, samples), np.zeros(samples)]
    v = np.r_[np.zeros(samples), np.linspace(0, h - 1, samples), np.full(samples, h - 1), np.linspace(0, h - 1, samples)]
    cx, cy = (w - 1) / 2.0, (h - 1) / 2.0

    def inside(f):
        """True if every border pixel of the output samples a pixel inside the raw image."""
        P = np.c_[(u - cx) / f, (v - cy) / f, np.ones(len(u))]
        uv, _ = cv2.projectPoints(P, np.zeros(3), np.zeros(3), K, dist)
        uv = uv.reshape(-1, 2)
        return bool(np.all((uv[:, 0] >= 0) & (uv[:, 0] <= w - 1) & (uv[:, 1] >= 0) & (uv[:, 1] <= h - 1)))

    lo, hi = 0.5 * K[0, 0], 3.0 * K[0, 0]
    for _ in range(40):
        mid = 0.5 * (lo + hi)
        lo, hi = (lo, mid) if inside(mid) else (mid, hi)
    return np.array([[hi, 0.0, cx], [0.0, hi, cy], [0.0, 0.0, 1.0]])


def write_undistort(path, K, dist, K_out, size, camera, stamp):
    """Undistortion file the avatar streamer reads: the calibrated camera and the pinhole to resample into."""
    import yaml
    w, h = int(size[0]), int(size[1])
    cam = {"width": w, "height": h, "fx": float(K[0, 0]), "fy": float(K[1, 1]), "cx": float(K[0, 2]),
           "cy": float(K[1, 2]), "dist": [float(v) for v in np.ravel(dist)[:5]]}
    out = {"width": w, "height": h, "fx": float(K_out[0, 0]), "fy": float(K_out[1, 1]), "cx": float(K_out[0, 2]),
           "cy": float(K_out[1, 2])}
    with open(path, "w") as f:
        yaml.safe_dump({"camera_name": camera, "date": stamp, "camera": cam, "output": out}, f, sort_keys=False)
    return path


def undistorted_pixels(uv, K, dist, K_out):
    """Where raw pixels land in the undistorted video."""
    return cv2.undistortPoints(np.asarray(uv, dtype=np.float64).reshape(-1, 1, 2), K, dist, None, K_out).reshape(-1, 2)


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
