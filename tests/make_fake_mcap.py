"""Builds a tiny FAKE mcap with the same topics as the challenge (simplified schemas) to test the
team scripts end-to-end before/without the 7 GiB file. Not part of the method."""
import math, sys
import numpy as np, cv2
from mcap_ros2.writer import Writer

SEP = "\n" + "=" * 80 + "\n"
TIME = "builtin_interfaces/Time"
HDR = "std_msgs/Header header\n" + SEP + "MSG: std_msgs/Header\nbuiltin_interfaces/Time stamp\nstring frame_id\n"
TF = ("geometry_msgs/TransformStamped[] transforms\n" + SEP + "MSG: geometry_msgs/TransformStamped\n"
      "std_msgs/Header header\nstring child_frame_id\ngeometry_msgs/Transform transform\n" + SEP +
      "MSG: std_msgs/Header\nbuiltin_interfaces/Time stamp\nstring frame_id\n" + SEP +
      "MSG: geometry_msgs/Transform\ngeometry_msgs/Vector3 translation\ngeometry_msgs/Quaternion rotation\n" + SEP +
      "MSG: geometry_msgs/Vector3\nfloat64 x\nfloat64 y\nfloat64 z\n" + SEP +
      "MSG: geometry_msgs/Quaternion\nfloat64 x\nfloat64 y\nfloat64 z\nfloat64 w\n")
IMG = HDR.replace("std_msgs/Header header\n", "std_msgs/Header header\nstring format\nuint8[] data\n")
INFO = HDR.replace("std_msgs/Header header\n", "std_msgs/Header header\nuint32 height\nuint32 width\nstring distortion_model\nfloat64[] d\nfloat64[9] k\n")
MARK = ("visualization_msgs/Marker[] markers\n" + SEP + "MSG: visualization_msgs/Marker\nstd_msgs/Header header\n"
        "string ns\nint32 id\nint32 type\ngeometry_msgs/Point[] points\n" + SEP +
        "MSG: std_msgs/Header\nbuiltin_interfaces/Time stamp\nstring frame_id\n" + SEP +
        "MSG: geometry_msgs/Point\nfloat64 x\nfloat64 y\nfloat64 z\n")
STATE = HDR.replace("std_msgs/Header header\n", "std_msgs/Header header\nfloat64 x\nfloat64 y\nfloat64 yaw\n"
                    "float64 vx\nfloat64 vy\nfloat64 yaw_rate\nfloat64 ax\nfloat64 ay\nfloat64 s\nfloat64 n\n")


def hdr(t, frame=""):
    return {"stamp": {"sec": t // 10**9, "nanosec": t % 10**9}, "frame_id": frame}


def quat_yaw(y):
    return {"x": 0.0, "y": 0.0, "z": math.sin(y / 2), "w": math.cos(y / 2)}


def main(out):
    T0 = 1_791_918_150_000_000_000
    th = np.linspace(0, 2 * np.pi, 2000, endpoint=False)
    R = 300 + 40 * np.cos(2 * th)
    C = np.stack([R * np.cos(th), R * np.sin(th)], 1)
    tg = np.gradient(C, axis=0); tg /= np.linalg.norm(tg, axis=1, keepdims=True)
    nrm = np.stack([-tg[:, 1], tg[:, 0]], 1)
    left, right = C + 6 * nrm, C - 6 * nrm
    pitC = C[100:300] + 25 * nrm[100:300]
    seg = np.r_[0, np.cumsum(np.linalg.norm(np.diff(np.vstack([C, C[:1]]), axis=0), axis=1))]
    L = seg[-1]
    # GT: 3 laps, start slightly before the gate (index 0), speed 30 m/s, lateral wiggle
    dt, v = 0.01, 30.0
    n_steps = int(3.2 * L / v / dt)
    s = (-50 + v * dt * np.arange(n_steps)) % L
    Cc = np.vstack([C, C[:1]]); Nc = np.vstack([nrm, nrm[:1]])
    Cs = np.stack([np.interp(s, seg, Cc[:, 0]), np.interp(s, seg, Cc[:, 1])], 1)
    Ns = np.stack([np.interp(s, seg, Nc[:, 0]), np.interp(s, seg, Nc[:, 1])], 1)
    Ns /= np.linalg.norm(Ns, axis=1, keepdims=True)
    i = np.searchsorted(seg, s) % len(C)
    lat = 2.0 * np.sin(2 * np.pi * s / L * 5)
    P = Cs + lat[:, None] * Ns
    yaw = np.unwrap(np.arctan2(np.gradient(P[:, 1]), np.gradient(P[:, 0])))
    yaw_rate = np.gradient(yaw) / dt
    v_true = np.linalg.norm(np.gradient(P, axis=0), axis=1) / dt
    with open(out, "wb") as f:
        w = Writer(f)
        sch = {k: w.register_msgdef(n, d) for k, n, d in [
            ("tf", "tf2_msgs/msg/TFMessage", TF), ("img", "sensor_msgs/msg/CompressedImage", IMG),
            ("info", "sensor_msgs/msg/CameraInfo", INFO), ("mark", "visualization_msgs/msg/MarkerArray", MARK),
            ("state", "sd_localization_msgs/msg/StateEstimation", STATE)]}
        cams = {"fl": 0, "fr": 0, "cl": 50, "cr": -50, "r": 180, "rl": 130, "rr": -130}
        jpg = cv2.imencode(".jpg", np.zeros((36, 75, 3), np.uint8))[1].tobytes()
        for k in range(n_steps):
            t = T0 + int(k * dt * 1e9)
            if k % 1000 == 0:
                trs = []
                for c, a in cams.items():
                    # optical frame: z forward. base->cam = Rz(a) * R_opt  (R_opt maps optical->base: z->x, x->-y, y->-z)
                    qa = (math.cos(math.radians(a) / 2), math.sin(math.radians(a) / 2))
                    q_opt = (0.5, -0.5, 0.5, -0.5)            # (x,y,z,w) rotation optical->base_link
                    w1, z1 = qa
                    x2, y2, z2, w2 = q_opt
                    q = (w1*x2 - z1*y2, w1*y2 + z1*x2, w1*z2 + z1*w2, w1*w2 - z1*z2)
                    trs.append({"header": hdr(t, "constructor0/base_link"), "child_frame_id": f"constructor0/camera_{c}",
                                "transform": {"translation": {"x": 1.0, "y": 0.0, "z": 0.8},
                                              "rotation": {"x": q[0], "y": q[1], "z": q[2], "w": q[3]}}})
                w.write_message("/tf_static", sch["tf"], {"transforms": trs}, t, t)
                mk = [{"header": hdr(t, "map"), "ns": n, "id": j, "type": 4,
                       "points": [{"x": float(p[0]), "y": float(p[1]), "z": 0.0} for p in arr]}
                      for j, (n, arr) in enumerate([("track_left", left), ("track_right", right),
                                                    ("pitlane_left", pitC + 4 * nrm[100:300]), ("pitlane_right", pitC - 4 * nrm[100:300])])]
                w.write_message("/constructor0/track_bounds", sch["mark"], {"markers": mk}, t, t)
            if k % 100 == 0:
                for c in cams:
                    w.write_message(f"/constructor0/sensor/camera_{c}/camera_info", sch["info"],
                                    {"header": hdr(t, f"constructor0/camera_{c}"), "height": 728, "width": 1506,
                                     "distortion_model": "plumb_bob", "d": [0.0] * 5,
                                     "k": [1350.0, 0, 753, 0, 1350.0, 364, 0, 0, 1]}, t, t)
            w.write_message("/tf", sch["tf"], {"transforms": [{"header": hdr(t, "map"), "child_frame_id": "constructor0/base_link",
                            "transform": {"translation": {"x": float(P[k, 0]), "y": float(P[k, 1]), "z": 0.0}, "rotation": quat_yaw(float(yaw[k]))}}]}, t, t)
            w.write_message("/constructor0/state_estimation", sch["state"], {"header": hdr(t, "map"),
                            "x": float(P[k, 0]), "y": float(P[k, 1]), "yaw": float(yaw[k]), "vx": float(v_true[k]), "vy": 0.0,
                            "yaw_rate": float(yaw_rate[k]), "ax": 0.0, "ay": float(v * yaw_rate[k]), "s": float(s[k]), "n": float(lat[k])}, t, t)
            if k % 10 == 0:
                for j, c in enumerate(cams):
                    te = t + j * 3_000_000                         # cameras 3 ms apart
                    w.write_message(f"/constructor0/sensor/camera_{c}/compressed_image", sch["img"],
                                    {"header": hdr(te, f"constructor0/camera_{c}"), "format": "jpeg", "data": jpg}, te + 20_000_000, te + 20_000_000)
        w.finish()
    print(f"fake mcap: {n_steps*dt:.0f} s, lap {L:.0f} m -> {out}")


if __name__ == "__main__":
    main(sys.argv[1])
