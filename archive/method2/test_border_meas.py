#!/usr/bin/env python3
"""DEV CHECK ONLY (reads ground truth) - is loc/border_meas.py good enough to feed the EKF?

    python loc/test_border_meas.py attitude                 # C1: map z / pitch / roll vs GT
    python loc/test_border_meas.py recover --cams cl cr fr  # C2: start 1 m off -> pulled back to GT?

Training laps 1-5 only (data/laps.csv); validation laps 6-7 are never touched here.
"""
import argparse, json, sys, time
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT)]
from archive.method2.border_meas import BorderMeasurer, MapGround, edge_dist  # noqa: E402


def train_window():
    L = pd.read_csv(ROOT / "data/laps.csv"); L = L[L.lap.isin([1, 2, 3, 4, 5])]
    return int(L.start_ns.min()), int(L.end_ns.max())


def gt_at(gt, t):
    tg = gt.timestamp_ns.to_numpy(np.int64)
    return (np.interp(t, tg, gt.x), np.interp(t, tg, gt.y), np.interp(t, tg, gt.z),
            np.angle(np.exp(1j * np.interp(t, tg, np.unwrap(gt.yaw)))))


def test_attitude(static, gt):
    t0, t1 = train_window()
    S = pd.read_csv(ROOT / "data/state_full.csv", usecols=["timestamp_ns", "pitch_rad", "roll_rad"])
    S = S[(S.timestamp_ns >= t0) & (S.timestamp_ns <= t1)].iloc[::100]          # 1 Hz
    t = S.timestamp_ns.to_numpy(np.int64)
    x, y, z, yaw = gt_at(gt, t)
    mg = MapGround(static)
    a = np.array([mg.attitude(*p) for p in zip(x, y, yaw)])
    ez = z - a[:, 0]
    ep, er = np.degrees(S.pitch_rad.to_numpy() - a[:, 1]), np.degrees(S.roll_rad.to_numpy() - a[:, 2])
    gp, gr = np.degrees(S.pitch_rad.to_numpy()), np.degrees(S.roll_rad.to_numpy())
    print(f"{len(t)} poses, laps 1-5")
    print(f"z    : GT - map  median {np.median(ez):+.3f} m, MAD {np.median(np.abs(ez - np.median(ez))):.3f} m")
    print(f"pitch: GT range {gp.min():+.2f}..{gp.max():+.2f} deg | error median {np.median(ep):+.2f}, "
          f"p95 |err| {np.percentile(np.abs(ep), 95):.2f} deg | |err| if pitch=0: {np.percentile(np.abs(gp), 95):.2f}")
    print(f"roll : GT range {gr.min():+.2f}..{gr.max():+.2f} deg | error median {np.median(er):+.2f}, "
          f"p95 |err| {np.percentile(np.abs(er), 95):.2f} deg | |err| if roll=0 : {np.percentile(np.abs(gr), 95):.2f}")
    print(f"corr(map, GT): pitch {np.corrcoef(a[:, 1], S.pitch_rad)[0, 1]:.3f}, roll {np.corrcoef(a[:, 2], S.roll_rad)[0, 1]:.3f}")


def test_recover(static, gt, cams, n, offsets, use_slope, z_offset):
    t0, t1 = train_window()
    idx = pd.read_csv(ROOT / "data/images/index.csv"); idx = idx[idx.file.notna()]
    ref = idx[(idx.cam == cams[0]) & (idx.stamp_ns >= t0) & (idx.stamp_ns <= t1)].sort_values("stamp_ns")
    S = pd.read_csv(ROOT / "data/state_full.csv", usecols=["timestamp_ns", "v_mps"])
    v = np.interp(ref.stamp_ns, S.timestamp_ns, S.v_mps)
    ref = ref[v > 8]
    ref = ref.iloc[np.linspace(0, len(ref) - 1, min(n, len(ref))).astype(int)]
    bm = BorderMeasurer(static, cams=cams, use_slope=use_slope, z_offset=z_offset)
    rows, t_meas = [], []
    for r in ref.itertuples():
        dists = {}
        for c in cams:      # same frame on every camera (hardware sync, |dt| < 0.05 ms)
            cand = idx[idx.cam == c]
            k = np.argmin(np.abs(cand.stamp_ns.to_numpy() - r.stamp_ns))
            img = cv2.imread(str(ROOT / "data/images" / cand.file.iloc[k]))
            dists[c] = edge_dist(img, c)
        gx, gy, _, gyaw = gt_at(gt, np.array([r.stamp_ns]))
        gx, gy, gyaw = gx[0], gy[0], gyaw[0]
        for off in offsets:
            pred = (gx - np.sin(gyaw) * off, gy + np.cos(gyaw) * off, gyaw)    # start `off` m to the left
            tic = time.perf_counter(); m = bm.measure(dists, pred); t_meas.append(time.perf_counter() - tic)
            if m is None:
                rows.append(dict(off=off, ok=False)); continue
            dx, dy = m["x"] - gx, m["y"] - gy
            rows.append(dict(off=off, ok=True, lat=-np.sin(gyaw) * dx + np.cos(gyaw) * dy,
                             yaw=np.degrees(m["yaw"] - gyaw), cost=m["cost"], contrast=m["contrast"],
                             n=m["n_pts"], edge=m["at_edge"]))
    D = pd.DataFrame(rows)
    print(f"cams {cams}, {len(ref)} frames x {len(offsets)} offsets, slope={'map' if use_slope else 'off'}, "
          f"{1000 * np.mean(t_meas):.0f} ms per measure")
    print(f"{'start off':>9s} {'meas %':>7s} {'|lat| med':>9s} {'<0.3 m %':>9s} {'<0.5 m %':>9s} {'yaw med':>8s} {'contrast':>8s}")
    for off, g in D.groupby("off"):
        o = g[g.ok]
        a = np.abs(o.lat)
        print(f"{off:+9.2f} {100 * len(o) / len(g):7.0f} {a.median():9.2f} {100 * (a < 0.3).mean():9.0f} "
              f"{100 * (a < 0.5).mean():9.0f} {o.yaw.abs().median():8.2f} {o.contrast.median():8.2f}")
    o = D[D.ok]
    good = o[(o.contrast > 1.0) & ~o.edge]
    print(f"with quality gate (contrast > 1 px, not at grid edge): keep {100 * len(good) / len(D):.0f} % | "
          f"|lat| median {good.lat.abs().median():.2f} m, <0.3 m {100 * (good.lat.abs() < 0.3).mean():.0f} %, "
          f">1 m (outliers) {100 * (good.lat.abs() > 1).mean():.0f} %")
    print(f"no-offset bias (0 m start): median lat {o[o.off == 0].lat.median():+.2f} m  (systematic calibration/height error)")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("test", choices=["attitude", "recover"])
    ap.add_argument("--cams", nargs="+", default=["cl", "cr", "fr"])
    ap.add_argument("--n", type=int, default=60)
    ap.add_argument("--offsets", type=float, nargs="+", default=[-1.0, -0.5, 0.0, 0.5, 1.0])
    ap.add_argument("--no-slope", action="store_true")
    ap.add_argument("--z-offset", type=float, default=0.0)
    a = ap.parse_args()
    static = json.load(open(ROOT / "data/static.json")); gt = pd.read_csv(ROOT / "data/gt.csv")
    if a.test == "attitude":
        test_attitude(static, gt)
    else:
        test_recover(static, gt, a.cams, a.n, a.offsets, not a.no_slope, a.z_offset)


if __name__ == "__main__":
    main()
