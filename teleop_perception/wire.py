import struct
import time

import msgpack

IDLE = 1
HOMING = 2
AWAITING = 3
ENGAGED = 4
POLICY = 0
HUMAN = 1
HOLD = 2

HEADER_FMT = "<IQQBBB"
ARM_CMD_FMT = HEADER_FMT + "3f4ffB"
ARM_STATE_FMT = HEADER_FMT + "3f4f7f7fBfBIII"
ARM_CMD_SIZE = struct.calcsize(ARM_CMD_FMT)
ARM_STATE_SIZE = struct.calcsize(ARM_STATE_FMT)

assert ARM_CMD_SIZE == 56, ARM_CMD_SIZE
assert ARM_STATE_SIZE == 125, ARM_STATE_SIZE


def pack_arm_command(seq, device_id, position, quaternion, gripper=0.0, state=ENGAGED):
    """Packs one ArmCommandMsg (common.hpp, pack(1)); quaternion is (w, x, y, z)."""
    now = time.time_ns()
    return struct.pack(ARM_CMD_FMT, seq & 0xFFFFFFFF, now, now, state, 0, device_id,
                       *[float(v) for v in position], *[float(v) for v in quaternion], float(gripper), 0)


def unpack_arm_state(data):
    """Decodes one ArmStateMsg into a dict, or None if the size does not match."""
    if len(data) != ARM_STATE_SIZE:
        return None
    f = struct.unpack(ARM_STATE_FMT, data)
    return {
        "seq": f[0], "timestamp_ns": f[1], "sample_time_ns": f[2], "state": f[3], "fault": f[4],
        "device_id": f[5], "position": f[6:9], "quaternion": f[9:13], "joints": f[13:20],
        "tau_ext": f[20:27], "recovering": bool(f[27]), "gripper_width": f[28], "grasp_state": f[29],
    }


def pack_envelope(seq, msg_type, payload, ack_requested=True, state=ENGAGED):
    """Packs one ReliableEnvelope (udp_reliable.hpp)."""
    return msgpack.packb({
        "sequence": seq & 0xFFFFFFFF, "timestamp_ns": time.time_ns(), "state": state, "fault_code": 0,
        "msg_type": msg_type, "ack_requested": ack_requested, "payload": payload,
    }, use_bin_type=True)
