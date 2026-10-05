import bisect
import socket
import threading
import time
from collections import deque

import msgpack

from .geometry import T_from_pos_quat, quat_from_R


class PosePublisher:
    """Sends fused object poses as one msgpack datagram per update."""

    def __init__(self, host, port):
        self.addr = (host, int(port))
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.seq = 0

    def send(self, estimates, capture_ns):
        """Publishes a list of WorldEstimate."""
        self.seq += 1
        objects = []
        for e in estimates:
            objects.append({
                "name": e.name,
                "position": [float(v) for v in e.T_world_obj[:3, 3]],
                "quaternion": [float(v) for v in quat_from_R(e.T_world_obj[:3, :3])],
                "n_tags": int(e.n_tags),
                "n_cams": len(e.cameras),
                "rms_px": float(e.rms_px),
                "static": bool(e.static),
                "ambiguous": bool(e.ambiguous),
            })
        msg = {"seq": self.seq, "timestamp_ns": int(capture_ns), "sent_ns": time.time_ns(), "objects": objects}
        self.sock.sendto(msgpack.packb(msg, use_bin_type=True), self.addr)

    def close(self):
        """Closes the socket."""
        self.sock.close()


class GroundTruthListener:
    """Receives the avatar's SceneObjectsMsg (sim object poses) and answers time lookups."""

    def __init__(self, bind_host, port, history=2000):
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self.sock.bind((bind_host, int(port)))
        self.sock.settimeout(0.2)
        self._times = deque(maxlen=history)
        self._poses = deque(maxlen=history)
        self._lock = threading.Lock()
        self._running = True
        self._thread = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()

    def _loop(self):
        """Receive thread."""
        while self._running:
            try:
                data, _ = self.sock.recvfrom(65535)
            except (socket.timeout, OSError):
                continue
            try:
                msg = msgpack.unpackb(data, raw=False, strict_map_key=False)
            except Exception:
                continue
            poses = {}
            for s in msg.get("slots", []):
                if len(s.get("position", [])) == 3 and len(s.get("quaternion", [])) == 4:
                    poses[s["name"]] = T_from_pos_quat(s["position"], s["quaternion"])
            with self._lock:
                self._times.append(int(msg.get("timestamp_ns", time.time_ns())))
                self._poses.append(poses)

    def lookup(self, t_ns, max_dt_ns):
        """Ground-truth poses closest in time to t_ns, or {} if none is close enough."""
        with self._lock:
            times = list(self._times)
            if not times:
                return {}
            k = bisect.bisect_left(times, t_ns)
            cands = [j for j in (k - 1, k) if 0 <= j < len(times)]
            j = min(cands, key=lambda j: abs(times[j] - t_ns))
            if abs(times[j] - t_ns) > max_dt_ns:
                return {}
            return self._poses[j]

    def close(self):
        """Stops the receive thread."""
        self._running = False
        self._thread.join(timeout=1.0)
        self.sock.close()


def unpack_poses(data):
    """Decodes one PosePublisher datagram into {name: 4x4}."""
    msg = msgpack.unpackb(data, raw=False)
    return {o["name"]: T_from_pos_quat(o["position"], o["quaternion"]) for o in msg["objects"]}, msg

