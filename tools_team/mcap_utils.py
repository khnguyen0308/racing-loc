"""Small helpers shared by the team scripts (reading the ROS 2 mcap of the challenge)."""
import math
from mcap.reader import make_reader
from mcap_ros2.decoder import DecoderFactory

TOPIC_PREFIX = "/constructor0"
CAMERAS = ["fl", "fr", "cl", "cr", "r", "rl", "rr"]


def img_topic(cam):
    return f"{TOPIC_PREFIX}/sensor/camera_{cam}/compressed_image"


def info_topic(cam):
    return f"{TOPIC_PREFIX}/sensor/camera_{cam}/camera_info"


def stamp_ns(header):
    return int(header.stamp.sec) * 10**9 + int(header.stamp.nanosec)


def iter_msgs(path, topics, start_ns=None, end_ns=None):
    """Yields (topic, log_time_ns, decoded_msg). start/end filter on the mcap log time."""
    with open(path, "rb") as f:
        r = make_reader(f, decoder_factories=[DecoderFactory()])
        for _, ch, m, msg in r.iter_decoded_messages(topics=topics, start_time=start_ns, end_time=end_ns):
            yield ch.topic, m.log_time, msg


def fields(msg):
    """Field names of a decoded ROS 2 message (works for mcap_ros2 dynamic classes)."""
    if hasattr(msg, "__slots__"):
        return [s for s in msg.__slots__ if not s.startswith("_")]
    if hasattr(msg, "__dict__"):
        return [k for k in vars(msg) if not k.startswith("_")]
    return []


def is_msg(x):
    return bool(fields(x)) and not isinstance(x, (int, float, str, bytes))


def to_dict(msg, max_list=4):
    """Recursive dict for printing; long lists/bytes are truncated."""
    if isinstance(msg, (bytes, bytearray)) or (isinstance(msg, (list, tuple)) and len(msg) and isinstance(msg[0], int) and len(msg) > 64):
        return f"<{len(msg)} bytes>"
    if isinstance(msg, (list, tuple)):
        out = [to_dict(v, max_list) for v in list(msg)[:max_list]]
        if len(msg) > max_list:
            out.append(f"... ({len(msg)} items)")
        return out
    if is_msg(msg):
        return {k: to_dict(getattr(msg, k), max_list) for k in fields(msg)}
    return msg


def flatten_numeric(msg, prefix=""):
    """{'a.b.c': float} for every numeric scalar leaf (lists of length <= 9 are expanded)."""
    out = {}
    if isinstance(msg, bool):
        out[prefix] = float(msg)
    elif isinstance(msg, (int, float)):
        out[prefix] = float(msg)
    elif isinstance(msg, (list, tuple)) and 0 < len(msg) <= 9 and all(isinstance(v, (int, float)) for v in msg):
        for i, v in enumerate(msg):
            out[f"{prefix}[{i}]"] = float(v)
    elif is_msg(msg):
        for k in fields(msg):
            out.update(flatten_numeric(getattr(msg, k), f"{prefix}.{k}" if prefix else k))
    return out


def quat_to_R(qx, qy, qz, qw):
    n = math.sqrt(qx*qx + qy*qy + qz*qz + qw*qw)
    qx, qy, qz, qw = qx/n, qy/n, qz/n, qw/n
    return [[1-2*(qy*qy+qz*qz), 2*(qx*qy-qz*qw),   2*(qx*qz+qy*qw)],
            [2*(qx*qy+qz*qw),   1-2*(qx*qx+qz*qz), 2*(qy*qz-qx*qw)],
            [2*(qx*qz-qy*qw),   2*(qy*qz+qx*qw),   1-2*(qx*qx+qy*qy)]]


def quat_to_yaw(qx, qy, qz, qw):
    return math.atan2(2 * (qw * qz + qx * qy), 1 - 2 * (qy * qy + qz * qz))
