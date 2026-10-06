import struct
import sys
import time
from dataclasses import dataclass

import cv2
import numpy as np

SHM_N_SLOTS = 3
SHM_MAX_W = 1280
SHM_MAX_H = 960
SHM_CHANNELS = 3
OFF_SLOTS = 16 + 8 * SHM_N_SLOTS
SLOT_SIZE = SHM_MAX_W * SHM_MAX_H * SHM_CHANNELS
BUFFER_SIZE = OFF_SLOTS + SHM_N_SLOTS * SLOT_SIZE


@dataclass
class Frame:
    camera: str
    gray: np.ndarray
    capture_ns: int
    seq: int


class _PosixShm:
    """Read-only view of a POSIX shared memory segment opened under its exact name (no prefix added)."""

    def __init__(self, name):
        import mmap
        import os
        import _posixshmem
        self._os = os
        fd = None
        for n in (name, "/" + name.lstrip("/")):
            try:
                fd = _posixshmem.shm_open(n, os.O_RDONLY, 0o600)
                break
            except OSError:
                continue
        if fd is None:
            raise FileNotFoundError(name)
        self.fd = fd
        self.size = os.fstat(fd).st_size
        self._mm = mmap.mmap(fd, self.size, prot=mmap.PROT_READ)
        self.buf = memoryview(self._mm)

    def close(self):
        """Unmaps and closes, never unlinks."""
        self.buf.release()
        self._mm.close()
        self._os.close(self.fd)


class Distorter:
    """Applies a known lens distortion to ideal pinhole images (sim ground-truth tests)."""

    def __init__(self, intr, iters=30):
        w, h = intr.width, intr.height
        K, d = intr.K, np.zeros(5)
        d[:len(np.ravel(intr.dist)[:5])] = np.ravel(intr.dist)[:5]
        k1, k2, p1, p2, k3 = d
        u, v = np.meshgrid(np.arange(w, dtype=np.float64), np.arange(h, dtype=np.float64))
        xd, yd = (u - K[0, 2]) / K[0, 0], (v - K[1, 2]) / K[1, 1]
        x, y = xd.copy(), yd.copy()
        for _ in range(iters):
            r2 = x * x + y * y
            rad = 1.0 + r2 * (k1 + r2 * (k2 + r2 * k3))
            x = (xd - (2.0 * p1 * x * y + p2 * (r2 + 2.0 * x * x))) / rad
            y = (yd - (p1 * (r2 + 2.0 * y * y) + 2.0 * p2 * x * y)) / rad
        self.map1, self.map2 = cv2.convertMaps((x * K[0, 0] + K[0, 2]).astype(np.float32),
                                               (y * K[1, 1] + K[1, 2]).astype(np.float32), cv2.CV_16SC2)

    def __call__(self, img):
        """Distorted version of an ideal image."""
        return cv2.remap(img, self.map1, self.map2, cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT)


class ShmSource:
    """Reads the newest frame from the simulator's SharedFrameBuffer (shared_memory.hpp)."""

    def __init__(self, camera, name, distort=None):
        self.camera = camera
        self.name = name
        self.distort = distort
        self._buf = None
        self._handle = None
        self._last_count = None
        self._next_try = 0.0

    def _open(self):
        """Attaches to an existing mapping; never creates one."""
        if sys.platform == "win32":
            import ctypes
            import ctypes.wintypes as wt
            k32 = ctypes.WinDLL("kernel32", use_last_error=True)
            k32.OpenFileMappingW.restype = wt.HANDLE
            k32.OpenFileMappingW.argtypes = [wt.DWORD, wt.BOOL, wt.LPCWSTR]
            k32.MapViewOfFile.restype = ctypes.c_void_p
            k32.MapViewOfFile.argtypes = [wt.HANDLE, wt.DWORD, wt.DWORD, wt.DWORD, ctypes.c_size_t]
            h = k32.OpenFileMappingW(0x0004, False, "Local\\" + self.name.lstrip("/"))
            if not h:
                return False
            ptr = k32.MapViewOfFile(h, 0x0004, 0, 0, BUFFER_SIZE)
            if not ptr:
                k32.CloseHandle(h)
                return False
            self._handle = (k32, h, ptr)
            self._buf = (ctypes.c_uint8 * BUFFER_SIZE).from_address(ptr)
            self._view = memoryview(self._buf).cast("B")
        else:
            try:
                self._buf = _PosixShm(self.name)
            except (FileNotFoundError, OSError):
                return False
            if self._buf.size < BUFFER_SIZE:
                self._buf.close()
                self._buf = None
                return False
            self._view = self._buf.buf
        return True

    def _header(self):
        """Reads write_idx, frame_count, width, height."""
        return struct.unpack_from("<IIII", self._view, 0)

    def read(self):
        """Returns the newest unseen frame, or None."""
        if self._buf is None:
            now = time.monotonic()
            if now < self._next_try:
                return None
            self._next_try = now + 1.0
            if not self._open():
                return None
        for _ in range(3):
            idx, count, w, h = self._header()
            if w == 0 or h == 0 or count == self._last_count:
                return None
            slot = (idx + SHM_N_SLOTS - 1) % SHM_N_SLOTS
            ts = struct.unpack_from("<Q", self._view, 16 + 8 * slot)[0]
            start = OFF_SLOTS + slot * SLOT_SIZE
            rgb = np.frombuffer(self._view[start:start + w * h * 3], dtype=np.uint8).reshape(h, w, 3)
            gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
            del rgb
            if self.distort is not None:
                gray = self.distort(gray)
            idx2 = self._header()[0]
            if (idx2 - idx) % (1 << 32) < SHM_N_SLOTS - 1:
                self._last_count = count
                return Frame(camera=self.camera, gray=gray, capture_ns=int(ts) or time.time_ns(), seq=int(count))
        return None

    def intrinsics(self):
        """Shared memory carries no intrinsics."""
        return None

    def close(self):
        """Detaches from the mapping."""
        if self._buf is None:
            return
        self._view.release()
        if sys.platform == "win32":
            k32, h, ptr = self._handle
            self._buf = None
            k32.UnmapViewOfFile(ptr)
            k32.CloseHandle(h)
        else:
            self._view = None
            self._buf.close()
        self._buf = None


class RealSenseSource:
    """Color stream of one RealSense camera with fixed exposure."""

    def __init__(self, camera, serial="", width=1280, height=720, fps=30, exposure_us=None, gain=None):
        import pyrealsense2 as rs
        self.camera = camera
        self._rs = rs
        self._pipe = rs.pipeline()
        cfg = rs.config()
        if serial:
            cfg.enable_device(str(serial))
        cfg.enable_stream(rs.stream.color, int(width), int(height), rs.format.bgr8, int(fps))
        self._profile = self._pipe.start(cfg)
        sensor = self._profile.get_device().first_color_sensor()
        if exposure_us:
            sensor.set_option(rs.option.enable_auto_exposure, 0)
            sensor.set_option(rs.option.exposure, float(exposure_us) / 100.0)
            if sensor.supports(rs.option.auto_exposure_priority):
                sensor.set_option(rs.option.auto_exposure_priority, 0)
        if gain is not None:
            sensor.set_option(rs.option.gain, float(gain))
        self._seq = 0

    def read(self):
        """Returns the next frame if one is ready, or None."""
        frames = self._pipe.poll_for_frames()
        if not frames:
            return None
        color = frames.get_color_frame()
        if not color:
            return None
        domain = color.get_frame_timestamp_domain()
        if domain in (self._rs.timestamp_domain.global_time, self._rs.timestamp_domain.system_time):
            ts = int(color.get_timestamp() * 1e6)
        else:
            ts = time.time_ns()
        gray = cv2.cvtColor(np.asanyarray(color.get_data()), cv2.COLOR_BGR2GRAY)
        self._seq += 1
        return Frame(camera=self.camera, gray=gray, capture_ns=ts, seq=self._seq)

    def intrinsics(self):
        """Factory color intrinsics as (width, height, K, dist)."""
        p = self._profile.get_stream(self._rs.stream.color).as_video_stream_profile().get_intrinsics()
        K = np.array([[p.fx, 0.0, p.ppx], [0.0, p.fy, p.ppy], [0.0, 0.0, 1.0]])
        return p.width, p.height, K, np.asarray(p.coeffs, dtype=float)

    def close(self):
        """Stops the pipeline."""
        self._pipe.stop()


def make_source(cam):
    """Creates the frame source a camera entry asks for."""
    s = dict(cam.source)
    kind = s.pop("type")
    if kind == "shm":
        distort = Distorter(cam.intrinsics_base or cam.intrinsics) if s.get("apply_distortion") else None
        return ShmSource(cam.name, s["name"], distort)
    if kind == "realsense":
        return RealSenseSource(cam.name, **s)
    raise ValueError(f"unknown source type '{kind}' for camera {cam.name}")
