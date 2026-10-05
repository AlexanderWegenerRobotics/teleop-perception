import os
from dataclasses import dataclass, field
from typing import Optional

import numpy as np
import yaml

from .geometry import T_from_spec, T_world_cam_look_at


@dataclass
class Tag:
    id: int
    size: float
    T: np.ndarray

    def corners(self):
        """Marker corners (TL, TR, BR, BL) in the parent frame, OpenCV order."""
        h = self.size / 2.0
        local = np.array([[-h, h, 0.0, 1.0], [h, h, 0.0, 1.0], [h, -h, 0.0, 1.0], [-h, -h, 0.0, 1.0]])
        return (self.T @ local.T).T[:, :3]


@dataclass
class ObjectModel:
    name: str
    layout: str
    tags: dict


@dataclass
class Board:
    name: str
    dictionary: str
    standoff: float
    objects: list
    layouts: dict

    def owner(self):
        """Map from tag id to object name."""
        return {tid: o.name for o in self.objects for tid in o.tags}


@dataclass
class Intrinsics:
    width: int
    height: int
    K: np.ndarray
    dist: np.ndarray


@dataclass
class CameraSpec:
    name: str
    source: dict
    intrinsics: Optional[Intrinsics]
    intrinsics_from_device: bool
    extrinsics_path: Optional[str]
    T_world_cam: Optional[np.ndarray] = None
    extrinsics_origin: str = "none"
    nominal: Optional[np.ndarray] = None
    nominal_look_at: Optional[list] = None


@dataclass
class Config:
    path: str
    mode: str
    board: Board
    cameras: list
    reference: dict
    reference_tags: list
    detection: dict
    network: dict
    calibration: dict
    logging: dict
    sim: dict
    raw: dict = field(default_factory=dict)


def _load(path):
    """Reads one YAML file."""
    with open(path, "r") as f:
        return yaml.safe_load(f) or {}


def _rel(base_file, p):
    """Resolves p relative to the directory of base_file."""
    if p is None or os.path.isabs(p):
        return p
    return os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(base_file)), p))


def _tag(spec, standoff, tid=None):
    """Builds a Tag, shifting its plane out along the face normal by the standoff."""
    T = T_from_spec(spec)
    T[:3, 3] = T[:3, 3] + standoff * T[:3, 2]
    return Tag(id=int(spec["id"] if tid is None else tid), size=float(spec["size"]), T=T)


def load_board(path):
    """Loads a board file: layouts plus object instances with their tag ids."""
    raw = _load(path)
    standoff = float(raw.get("tag_standoff", 0.0))
    layouts = raw["layouts"]
    objects, seen = [], {}
    for o in raw["objects"]:
        slots = layouts[o["layout"]]["tags"]
        ids = o["ids"]
        if len(ids) != len(slots):
            raise ValueError(f"{o['name']}: {len(ids)} ids for {len(slots)} tag slots in layout {o['layout']}")
        tags = {}
        for tid, slot in zip(ids, slots):
            if tid in seen:
                raise ValueError(f"tag id {tid} used by {seen[tid]} and {o['name']}")
            seen[tid] = o["name"]
            tags[int(tid)] = _tag(slot, standoff, tid)
        objects.append(ObjectModel(name=o["name"], layout=o["layout"], tags=tags))
    return Board(name=raw["name"], dictionary=raw["dictionary"], standoff=standoff, objects=objects, layouts=layouts)


def load_intrinsics(path):
    """Loads intrinsics from explicit K/dist or from a sim vertical field of view."""
    raw = _load(path)
    w, h = int(raw["width"]), int(raw["height"])
    if "fovy_deg" in raw:
        fy = (h / 2.0) / np.tan(np.radians(float(raw["fovy_deg"])) / 2.0)
        K = np.array([[fy, 0.0, (w - 1) / 2.0], [0.0, fy, (h - 1) / 2.0], [0.0, 0.0, 1.0]])
        dist = np.zeros(5)
    else:
        K = np.array([[raw["fx"], 0.0, raw["cx"]], [0.0, raw["fy"], raw["cy"]], [0.0, 0.0, 1.0]], dtype=float)
        dist = np.asarray(raw.get("dist", [0.0] * 5), dtype=float)
    return Intrinsics(width=w, height=h, K=K, dist=dist)


def load_extrinsics(path):
    """Loads a world-from-camera transform written by calibration, or None."""
    if not path or not os.path.exists(path):
        return None
    raw = _load(path)
    return np.asarray(raw["T_world_cam"], dtype=float).reshape(4, 4)


def load_cameras(path, mode):
    """Loads the camera list for one mode and resolves intrinsics and extrinsics."""
    raw = _load(path)
    cams = []
    for c in raw["cameras"]:
        from_device = c.get("intrinsics") == "device"
        intr = None if from_device else load_intrinsics(_rel(path, c["intrinsics"]))
        ext_path = _rel(path, c.get("extrinsics", "").format(mode=mode)) if c.get("extrinsics") else None
        nominal = None
        if "nominal" in c:
            nominal = T_world_cam_look_at(c["nominal"]["pos"], c["nominal"]["look_at"])
        cam = CameraSpec(name=c["name"], source=dict(c["source"]), intrinsics=intr,
                         intrinsics_from_device=from_device, extrinsics_path=ext_path, nominal=nominal,
                         nominal_look_at=c["nominal"]["look_at"] if "nominal" in c else None)
        T = load_extrinsics(ext_path)
        if T is not None:
            cam.T_world_cam, cam.extrinsics_origin = T, "calibration"
        elif nominal is not None:
            cam.T_world_cam, cam.extrinsics_origin = nominal, "nominal"
        cams.append(cam)
    return cams


def load_reference(path, mode, standoff):
    """Loads the reference tags and, if calibration ran, where each camera saw them right afterwards."""
    raw = _load(path)
    tags = {int(s["id"]): _tag(s, standoff) for s in raw.get("tags", [])}
    measured = _rel(path, raw.get("measured", "").format(mode=mode)) if raw.get("measured") else None
    raw["observations"] = {}
    raw["origin"] = "nominal"
    if measured and os.path.exists(measured):
        obs = _load(measured).get("cameras", {})
        raw["observations"] = {cam: {int(t): {"corners": np.asarray(v["corners"], dtype=float),
                                               "depth_m": float(v["depth_m"])} for t, v in d.items()}
                               for cam, d in obs.items()}
        if raw["observations"]:
            raw["origin"] = "observed after calibration"
    raw["measured_path"] = measured
    return raw, list(tags.values())


def load_config(path, mode=None):
    """Loads the master config and every file it points to."""
    path = os.path.abspath(path)
    raw = _load(path)
    mode = mode or raw["mode"]
    if mode not in raw["cameras"]:
        raise ValueError(f"mode '{mode}' has no camera file in {path}")
    board = load_board(_rel(path, raw["board"]))
    cameras = load_cameras(_rel(path, raw["cameras"][mode]), mode)
    reference, ref_tags = load_reference(_rel(path, raw["reference"]), mode, board.standoff)
    clash = set(board.owner()) & {t.id for t in ref_tags}
    if clash:
        raise ValueError(f"reference tag ids {sorted(clash)} are also used on objects")
    calib = _load(_rel(path, raw["calibration"]))
    sim = dict(raw.get("sim", {}))
    if "root" in sim:
        sim["root"] = _rel(path, sim["root"])
    logging = dict(raw.get("logging", {}))
    if "dir" in logging:
        logging["dir"] = _rel(path, logging["dir"])
    return Config(
        path=path, mode=mode, board=board, cameras=cameras, reference=reference, reference_tags=ref_tags,
        detection=_load(_rel(path, raw["detection"])), network=_load(_rel(path, raw["network"])),
        calibration=calib, logging=logging, sim=sim, raw=raw,
    )


def calib_dir(cfg):
    """Directory where calibration results are written."""
    for cam in cfg.cameras:
        if cam.extrinsics_path:
            return os.path.dirname(cam.extrinsics_path)
    return os.path.normpath(os.path.join(os.path.dirname(cfg.path), "..", "calib"))
