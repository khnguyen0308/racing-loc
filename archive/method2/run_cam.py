#!/usr/bin/env python3
"""Causal EKF + camera border measurement -> pose CSV at camera rate (INTRO method 2 + 4).

    python loc/run_cam.py --gt data/gt_train.csv --out out/cam_train.csv      # develop on laps 1-5
    python loc/run_cam.py --gt data/gt_val.csv   --out out/cam_val.csv        # score on laps 6-7

Per camera frame t: predict with speed + yaw rate (stamps <= t), and if images of frame t exist,
match the map borders around the PREDICTED pose and update the lateral position.
--gt is used ONLY for the first pose and the time window (as run_ekf.py).
Images: the JPEGs of tools_team/extract_images.py (every 10th frame = 1 Hz camera updates).
"""
import argparse, json, sys, time
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT), str(ROOT / "tools_team")]
from archive.method2.border_meas import BorderMeasurer, edge_dist  # noqa: E402
from loc.ekf import CarEKF, body_cov, wrap              # noqa: E402
from state_io import load_state                         # noqa: E402


def jpeg_frames(images, cams, ref="fl"):
    """{fl stamp: {cam: path}} for frames saved on all cams (hardware sync, |dt| < 1 ms)."""
    idx = pd.read_csv(Path(images) / "index.csv"); idx = idx[idx.file.notna()]
    out = {}
    base = idx[idx.cam == ref]
    for c in cams:
        s = idx[idx.cam == c].sort_values("stamp_ns")
        st = s.stamp_ns.to_numpy(np.int64)
        k = np.clip(np.searchsorted(st, base.stamp_ns.to_numpy(np.int64)), 1, len(st) - 1)
        k = np.where(np.abs(st[k - 1] - base.stamp_ns.values) < np.abs(st[k] - base.stamp_ns.values), k - 1, k)
        for t, j in zip(base.stamp_ns.to_numpy(np.int64), k):
            if abs(st[j] - t) < 1_000_000:
                out.setdefault(int(t), {})[c] = str(Path(images) / s.file.iloc[j])
    return {t: v for t, v in out.items() if len(v) == len(cams)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--state", default="data/state_full.csv"); ap.add_argument("--gt", required=True)
    ap.add_argument("--stamps", default="data/images/index.csv"); ap.add_argument("--images", default="data/images")
    ap.add_argument("--static", default="data/static.json"); ap.add_argument("--out", required=True)
    ap.add_argument("--cams", nargs="+", default=["cl", "cr", "fr"])
    ap.add_argument("--sig-lat", type=float, default=0.8, help="m, lateral std of one measurement")
    ap.add_argument("--min-contrast", type=float, default=0.5, help="px, reject flat cost profiles")
    ap.add_argument("--trunc", type=float, default=8.0)
    ap.add_argument("--no-camera", action="store_true", help="prediction only (= B0 through this loop)")
    a = ap.parse_args()

    gt = pd.read_csv(a.gt)
    t0, t1 = int(gt.timestamp_ns.iloc[0]), int(gt.timestamp_ns.iloc[-1])
    S = load_state(a.state, ["v_mps", "wz_radps"])
    ts, v, wz = S.timestamp_ns.to_numpy(np.int64), S.v_mps.to_numpy(float), S.wz_radps.to_numpy(float)
    idx = pd.read_csv(a.stamps)
    t_out = np.sort(idx[idx.cam == "fl"].stamp_ns.to_numpy(np.int64)); t_out = t_out[(t_out >= t0) & (t_out <= t1)]
    frames = {} if a.no_camera else jpeg_frames(a.images, a.cams)
    bm = BorderMeasurer(json.load(open(a.static)), cams=a.cams, lat=np.arange(-1.5, 1.501, 0.05),
                        yaw_deg=np.array([0.0]), trunc=a.trunc)

    f = CarEKF(gt.x.iloc[0], gt.y.iloc[0], gt.yaw.iloc[0])      # first pose: the only GT used
    j = max(np.searchsorted(ts, t0, side="right") - 1, 0); t_cur = t0
    out, dbg, t_meas = [], [], []
    for t in t_out:
        while j + 1 < len(ts) and ts[j + 1] <= t:                 # causal predict up to t
            f.predict(v[j], wz[j], (ts[j + 1] - t_cur) / 1e9); t_cur = ts[j + 1]; j += 1
        f.predict(v[j], wz[j], (t - t_cur) / 1e9); t_cur = t
        if int(t) in frames:
            tic = time.perf_counter()
            dists = {c: edge_dist(cv2.imread(p), c) for c, p in frames[int(t)].items()}
            m = bm.measure(dists, tuple(f.X[:3]))
            t_meas.append(time.perf_counter() - tic)
            used = False
            if m is not None and m["contrast"] >= a.min_contrast and not m["at_edge"]:
                yaw = f.X[2]                                      # lateral-only: yaw/along-track ~free
                used, _ = f.update_pose((m["x"], m["y"], yaw), body_cov(yaw, 1e3, a.sig_lat, 1e3))
            dbg.append(dict(timestamp_ns=t, dlat=np.nan if m is None else m["dlat"],
                            contrast=np.nan if m is None else m["contrast"], used=used))
        out.append((t, f.X[0], f.X[1], wrap(f.X[2])))
    pd.DataFrame(out, columns=["timestamp_ns", "x", "y", "yaw"]).to_csv(a.out, index=False, float_format="%.6f")
    pd.DataFrame(dbg).to_csv(a.out.replace(".csv", "_meas.csv"), index=False)
    n_used = sum(d["used"] for d in dbg)
    ms = 1000 * np.mean(t_meas) if t_meas else 0.0
    print(f"{len(out)} poses -> {a.out} | camera frames {len(dbg)}, used {n_used}, rejected/flat {len(dbg) - n_used} "
          f"| {ms:.0f} ms per camera frame | final bias {f.X[3]:+.6f} rad/s, k_v {f.X[4]:.4f}")


if __name__ == "__main__":
    main()
