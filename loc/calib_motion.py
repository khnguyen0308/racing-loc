#!/usr/bin/env python3
"""Calibrate the motion model on TRAINING laps (GT allowed here: this is training, not inference).

    python loc/calib_motion.py --laps 1 2 3 4          # dev  (test on lap 5)
    python loc/calib_motion.py --laps 1 2 3 4 5        # final (validate on laps 6-7)

yaw-rate bias  b  = (integral of wz dt - GT heading change) / T     -> true yaw rate = wz - b
speed scale    k  = GT path length / integral of v dt               -> true speed    = k * v
Writes config/motion_calib.json; b0_dead_reckoning.py --yawrate-bias / --speed-scale and the EKF use it.
"""
import argparse, json
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]


def calibrate(gt, S, laps, speed_col="v_mps"):
    """Per-lap and pooled (b, k). gt: GT DataFrame, S: state DataFrame, laps: rows of laps.csv."""
    res = []
    for r in laps.itertuples():
        g = gt[(gt.timestamp_ns >= r.start_ns) & (gt.timestamp_ns <= r.end_ns)]
        s = S[(S.timestamp_ns >= r.start_ns) & (S.timestamp_ns <= r.end_ns)]
        ts = s.timestamp_ns.to_numpy(np.int64) / 1e9; dt = np.diff(ts)
        T = ts[-1] - ts[0]
        dyaw_gt = np.unwrap(g.yaw.to_numpy())[-1] - np.unwrap(g.yaw.to_numpy())[0]
        b = (np.sum(s.wz_radps.to_numpy()[:-1] * dt) - dyaw_gt) / T
        path_gt = np.sum(np.hypot(np.diff(g.x.to_numpy()), np.diff(g.y.to_numpy())))
        k = path_gt / np.sum(s[speed_col].to_numpy()[:-1] * dt)
        res.append(dict(lap=r.lap, T=T, b=b, k=k))
    D = pd.DataFrame(res)
    w = D["T"] / D["T"].sum()                     # pool weighted by lap duration
    return D, float((D.b * w).sum()), float((D.k * w).sum())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--laps", type=int, nargs="+", required=True)
    ap.add_argument("--speed-col", default="v_mps")
    ap.add_argument("--out", default=str(ROOT / "config/motion_calib.json"))
    ap.add_argument("--allow-val-laps", action="store_true", help="final calibration for the hidden test run only")
    a = ap.parse_args()
    if set(a.laps) & {6, 7} and not a.allow_val_laps:
        raise SystemExit("laps 6 and 7 are validation: never calibrate on them (--allow-val-laps: test run only)")
    L = pd.read_csv(ROOT / "data/laps.csv"); L = L[L.lap.isin(a.laps)]
    gt = pd.read_csv(ROOT / "data/gt.csv")
    S = pd.read_csv(ROOT / "data/state_full.csv", usecols=["timestamp_ns", "wz_radps", a.speed_col]).sort_values("timestamp_ns")
    D, b, k = calibrate(gt, S, L, a.speed_col)
    for r in D.itertuples():
        print(f"lap {r.lap}: {r.T:6.1f} s  bias {r.b:+.6f} rad/s  speed scale {r.k:.5f}")
    print(f"pooled laps {a.laps}: bias {b:+.6f} rad/s, speed scale {k:.5f} (std over laps: {D.b.std():.6f}, {D.k.std():.5f})")
    json.dump(dict(laps=a.laps, speed_col=a.speed_col, yawrate_bias=b, speed_scale=k), open(a.out, "w"), indent=1)
    print(f"-> {a.out}")


if __name__ == "__main__":
    main()
