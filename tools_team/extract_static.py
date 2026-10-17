#!/usr/bin/env python3
"""STEP 2 - static data: camera intrinsics (K, D), extrinsics base_link->camera (4x4) and the
track borders (track_left/right, pitlane_left/right) -> one JSON used by everybody.

    python3 tools_team/extract_static.py data/c0_..._camloc_0.mcap data/static.json
"""
import json, sys
from pathlib import Path
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from tools_team.mcap_utils import CAMERAS, info_topic, iter_msgs, quat_to_R, stamp_ns  # noqa: E402

TF_STATIC, BOUNDS = "/tf_static", "/constructor0/track_bounds"


def T_from_tf(tr):
    T = np.eye(4)
    q, p = tr.rotation, tr.translation
    T[:3, :3] = quat_to_R(q.x, q.y, q.z, q.w)
    T[:3, 3] = [p.x, p.y, p.z]
    return T


def chain(edges, parent, child):
    """Compose static transforms parent->...->child (edges: {(parent, child): T})."""
    stack = [(parent, np.eye(4))]
    seen = {parent}
    while stack:
        f, T = stack.pop()
        if f == child:
            return T
        for (p, c), Tpc in edges.items():
            if p == f and c not in seen:
                seen.add(c); stack.append((c, T @ Tpc))
    return None


def main(path, out):
    topics = [TF_STATIC, BOUNDS] + [info_topic(c) for c in CAMERAS]
    edges, infos, bounds = {}, {}, {}
    for topic, _, msg in iter_msgs(path, topics):
        if topic == TF_STATIC:
            for t in msg.transforms:
                edges[(t.header.frame_id, t.child_frame_id)] = T_from_tf(t.transform)
        elif topic == BOUNDS:
            for m in msg.markers:
                name = m.ns or str(m.id)
                bounds[name] = {"frame_id": m.header.frame_id, "type": int(m.type),
                                "points": [[p.x, p.y, p.z] for p in m.points]}
        else:
            cam = topic.split("camera_")[1].split("/")[0]
            infos.setdefault(cam, {"frame_id": msg.header.frame_id, "width": int(msg.width),
                                   "height": int(msg.height), "distortion_model": msg.distortion_model,
                                   "K": np.array(msg.k, float).reshape(3, 3).tolist(),
                                   "D": list(map(float, msg.d)), "stamp_ns": stamp_ns(msg.header)})
        if len(infos) == len(CAMERAS) and bounds and edges:
            break
    base = "constructor0/base_link"
    cams = {}
    for cam, inf in infos.items():
        frame = inf["frame_id"] or f"constructor0/camera_{cam}"
        T = chain(edges, base, frame)
        if T is None:
            print(f"WARNING: no tf {base} -> {frame}")
        cams[cam] = {**inf, "T_base_cam": None if T is None else T.tolist()}
        if T is not None:
            z = T[:3, :3] @ [0, 0, 1]     # optical axis in base_link
            print(f"cam {cam:2s} {inf['width']}x{inf['height']} fx={inf['K'][0][0]:7.1f} "
                  f"pos=({T[0,3]:+.2f},{T[1,3]:+.2f},{T[2,3]:+.2f}) m  looks at yaw "
                  f"{np.degrees(np.arctan2(z[1], z[0])):+7.1f} deg, pitch {np.degrees(np.arcsin(-z[2])):+5.1f} deg down")
    for k, b in bounds.items():
        p = np.array(b["points"])[:, :2]
        L = np.sum(np.linalg.norm(np.diff(p, axis=0), axis=1)) if len(p) > 1 else 0
        print(f"bound {k:15s} {len(p):6d} pts, length {L:8.1f} m, frame {b['frame_id']}")
    json.dump({"cameras": cams, "track_bounds": bounds,
               "tf_static": {f"{p}->{c}": T.tolist() for (p, c), T in edges.items()}},
              open(out, "w"), indent=1)
    print(f"-> {out}")


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])