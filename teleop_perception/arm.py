import socket
import threading
import time
from collections import deque

import cv2
import msgpack
import numpy as np

from . import wire
from .geometry import T_from_pos_quat, average_poses, make_T, pose_error, quat_from_R


def _slerp_R(R0, R1, s):
    """Rotation a fraction s of the way from R0 to R1."""
    rv, _ = cv2.Rodrigues(R0.T @ R1)
    Rs, _ = cv2.Rodrigues(rv * s)
    return R0 @ Rs


class ArmClient:
    """Drives one arm through the avatar's absolute (POLICY) channel and reads its measured pose."""

    def __init__(self, arm_cfg, motion_cfg):
        self.ip = arm_cfg["avatar_ip"]
        self.device = arm_cfg["device"]
        self.device_id = int(arm_cfg["device_id"])
        self.abs_port = int(arm_cfg["abs_cmd_port"])
        self.cmd_port = int(arm_cfg["cmd_port"])
        self.dt = 1.0 / float(arm_cfg.get("rate_hz", 100))
        self.lin_speed = float(motion_cfg.get("lin_speed", 0.05))
        self.rot_speed = np.radians(float(motion_cfg.get("rot_speed_deg", 15.0)))
        self.state_sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.state_sock.bind(("0.0.0.0", int(arm_cfg["state_port"])))
        self.state_sock.settimeout(0.1)
        self.cmd_sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.rel_sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.rel_sock.settimeout(0.3)
        self.seq = int(time.time() * 1000) & 0x7FFFFFFF
        self.rel_seq = self.seq
        self.history = deque(maxlen=2000)
        self.lock = threading.Lock()
        self.cmd_T = None
        self.goal_T = None
        self.running = True
        self.streaming = False
        self.rx = threading.Thread(target=self._recv_loop, daemon=True)
        self.rx.start()
        self.tx = threading.Thread(target=self._send_loop, daemon=True)
        self.heartbeat = False
        self.hb = None
        self.max_gap = 0.0
        self.last_send_t = None
        self.send_errors = 0
        self.last_send_error = None

    def _recv_loop(self):
        """Collects ArmStateMsg packets."""
        while self.running:
            try:
                data, _ = self.state_sock.recvfrom(256)
            except (socket.timeout, OSError):
                continue
            s = wire.unpack_arm_state(data)
            if s is None:
                continue
            T = T_from_pos_quat(s["position"], s["quaternion"])
            with self.lock:
                self.history.append((time.monotonic(), T, s))

    def latest(self):
        """Newest (time, T_world_ee, state dict), or None."""
        with self.lock:
            return self.history[-1] if self.history else None

    def wait_for_state(self, timeout=5.0):
        """Blocks until the arm state stream is alive."""
        t_end = time.monotonic() + timeout
        while time.monotonic() < t_end:
            s = self.latest()
            if s is not None and time.monotonic() - s[0] < 0.5:
                return s
            time.sleep(0.05)
        raise TimeoutError(f"no ArmStateMsg on port {self.state_sock.getsockname()[1]} -- is the avatar running "
                           f"and transmission_absolute configured for {self.device}?")

    def _send_rel(self, msg_type, payload, ack_requested=False):
        """Sends one envelope on the avatar's cmd channel."""
        with self.lock:
            self.rel_seq += 1
            seq = self.rel_seq
        self.rel_sock.sendto(wire.pack_envelope(seq, msg_type, payload, ack_requested), (self.ip, self.cmd_port))

    def _heartbeat_loop(self):
        """Keeps the avatar's deadman switch satisfied while this client owns the session."""
        t0 = time.monotonic()
        while self.running and self.heartbeat:
            self._send_rel("heartbeat", {"uptime_ms": int((time.monotonic() - t0) * 1000)})
            time.sleep(0.5)

    def arm_state(self):
        """SysState from the newest ArmStateMsg header, or None."""
        s = self.latest()
        return None if s is None else s[2]["state"]

    def _wait_state(self, state, timeout):
        """Waits until the arm reports the given SysState."""
        t_end = time.monotonic() + timeout
        while time.monotonic() < t_end:
            if self.arm_state() == state:
                return True
            time.sleep(0.05)
        return False

    def engage(self, homing_timeout=60.0, engage_timeout=10.0):
        """Walks IDLE -> HOMING -> AWAITING -> ENGAGED like the orchestrator; returns True if it engaged."""
        self.wait_for_state()
        self.heartbeat = True
        self.hb = threading.Thread(target=self._heartbeat_loop, daemon=True)
        self.hb.start()
        if self.arm_state() == wire.ENGAGED:
            return False
        self._send_rel("state_change", {"requested_state": wire.HOMING})
        if not self._wait_state(wire.AWAITING, homing_timeout):
            raise RuntimeError(f"arm did not reach AWAITING after homing (state {self.arm_state()})")
        self._send_rel("state_change", {"requested_state": wire.ENGAGED})
        if not self._wait_state(wire.ENGAGED, engage_timeout):
            raise RuntimeError(f"arm did not engage (state {self.arm_state()})")
        return True

    def disengage(self):
        """Requests IDLE."""
        self._send_rel("state_change", {"requested_state": wire.IDLE})

    def request_authority(self, authority, source="perception_calib", retries=5):
        """Sends an authority_request and waits for the avatar's ack."""
        payload = {"authority": int(authority), "source": source, "device": self.device}
        for _ in range(retries):
            self._send_rel("authority_request", payload, ack_requested=True)
            try:
                data, _ = self.rel_sock.recvfrom(4096)
                env = msgpack.unpackb(data, raw=False, strict_map_key=False)
                if env.get("msg_type") == "ack":
                    return True
            except (socket.timeout, OSError, ValueError):
                continue
        return False

    def _send_loop(self):
        """Streams the interpolated command at a fixed rate so the authority watchdog stays quiet."""
        next_t = prev_t = time.monotonic()
        while self.running and self.streaming:
            with self.lock:
                cmd, goal = self.cmd_T, self.goal_T
            t = time.monotonic()
            step, prev_t = min(t - prev_t, 5.0 * self.dt), t
            if goal is not None and step > 0.0:
                dp = goal[:3, 3] - cmd[:3, 3]
                _, dr_deg = pose_error(cmd, goal)
                dr = np.radians(dr_deg)
                n = max(np.linalg.norm(dp) / (self.lin_speed * step), dr / (self.rot_speed * step), 1.0)
                s = 1.0 / n
                cmd = make_T(_slerp_R(cmd[:3, :3], goal[:3, :3], s), cmd[:3, 3] + s * dp)
                with self.lock:
                    self.cmd_T = cmd
            self.seq += 1
            try:
                self.cmd_sock.sendto(wire.pack_arm_command(self.seq, self.device_id, cmd[:3, 3],
                                                           quat_from_R(cmd[:3, :3])), (self.ip, self.abs_port))
            except OSError as e:
                self.send_errors += 1
                self.last_send_error = str(e)
            now = time.monotonic()
            if self.last_send_t is not None:
                self.max_gap = max(self.max_gap, now - self.last_send_t)
            self.last_send_t = now
            next_t = max(next_t + self.dt, now - self.dt)
            time.sleep(max(0.0, next_t - time.monotonic()))

    def pop_stream_stats(self):
        """Largest gap between two sent commands (s) and send errors since the last call."""
        gap, err, msg = self.max_gap, self.send_errors, self.last_send_error
        self.max_gap, self.send_errors, self.last_send_error = 0.0, 0, None
        return gap, err, msg

    def start_streaming(self):
        """Starts commanding the measured pose, so nothing moves until move_to."""
        _, T, _ = self.wait_for_state()
        with self.lock:
            self.cmd_T = T.copy()
            self.goal_T = T.copy()
        self.streaming = True
        self.tx.start()

    def move_to(self, T_goal, timeout=60.0):
        """Ramps the command to T_goal at the configured speeds and waits until the command arrives."""
        with self.lock:
            self.goal_T = T_goal.copy()
        t_end = time.monotonic() + timeout
        while time.monotonic() < t_end:
            with self.lock:
                dp, dr = pose_error(self.cmd_T, self.goal_T)
            if dp < 1e-5 and dr < 1e-3:
                return True
            time.sleep(0.02)
        return False

    def wait_settled(self, window_s, pos_tol, rot_tol_deg, timeout_s, goal=None, goal_tol=(0.02, 3.0)):
        """Waits until the measured pose is near the goal and stays still for window_s; returns the mean pose or None."""
        t_end = time.monotonic() + timeout_s
        while time.monotonic() < t_end:
            time.sleep(0.05)
            now = time.monotonic()
            with self.lock:
                recent = [h for h in self.history if now - h[0] <= window_s]
                span = (now - self.history[0][0]) if self.history else 0.0
            if span < window_s or len(recent) < 3:
                continue
            ref = recent[-1][1]
            if goal is not None:
                gp, gr = pose_error(ref, goal)
                if gp > goal_tol[0] or gr > goal_tol[1]:
                    continue
            if all(pose_error(h[1], ref)[0] <= pos_tol and pose_error(h[1], ref)[1] <= rot_tol_deg for h in recent):
                return average_poses([h[1] for h in recent])
        return None

    def measured_since(self, t0):
        """Mean measured pose of the samples received after monotonic time t0."""
        with self.lock:
            Ts = [h[1] for h in self.history if h[0] >= t0]
        return average_poses(Ts) if Ts else None

    def close(self, release=True):
        """Stops streaming and optionally hands the arm back to HOLD."""
        self.streaming = False
        if self.tx.is_alive():
            self.tx.join(timeout=1.0)
        if release:
            self.request_authority(wire.HOLD)
        self.heartbeat = False
        if self.hb is not None:
            self.hb.join(timeout=1.0)
        self.running = False
        self.rx.join(timeout=1.0)
        for s in (self.state_sock, self.cmd_sock, self.rel_sock):
            s.close()

