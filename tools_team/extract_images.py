#!/usr/bin/env python3
"""STEP 4 - image index (+ optional JPEG dump) for development.

Writes data/images/index.csv: cam, stamp_ns (= exposure time), log_time_ns, file
With --save-every N only every N-th frame of the chosen cameras is written to disk
(the full run is ~83k images, 7 GiB -> save what you need, read the mcap directly in run.py).

    python3 tools_team/extract_images.py data/c0.mcap data/images --cams fl fr cl cr --save-every 10
    python3 tools_team/extract_images.py data/c0.mcap data/images --index-only
"""
import argparse, csv, os, sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from tools_team.mcap_utils import CAMERAS, img_topic, iter_msgs, stamp_ns  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("mcap"); ap.add_argument("out")
    ap.add_argument("--cams", nargs="+", default=CAMERAS)
    ap.add_argument("--save-every", type=int, default=10)
    ap.add_argument("--index-only", action="store_true")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    count = {c: 0 for c in a.cams}
    with open(os.path.join(a.out, "index.csv"), "w", newline="") as f:
        w = csv.writer(f); w.writerow(["cam", "stamp_ns", "log_time_ns", "file"])
        for topic, log_ns, msg in iter_msgs(a.mcap, [img_topic(c) for c in a.cams]):
            cam = topic.split("camera_")[1].split("/")[0]
            t = stamp_ns(msg.header); fn = ""
            if not a.index_only and count[cam] % a.save_every == 0:
                os.makedirs(os.path.join(a.out, cam), exist_ok=True)
                fn = f"{cam}/{t}.jpg"
                with open(os.path.join(a.out, fn), "wb") as g:
                    g.write(bytes(msg.data))
            count[cam] += 1
            w.writerow([cam, t, log_ns, fn])
    print("frames per camera:", count)


if __name__ == "__main__":
    main()
