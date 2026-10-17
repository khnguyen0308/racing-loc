#!/usr/bin/env python3
"""BASELINE B0 - dead reckoning from the first pose with speed + yaw rate of the car state.
No camera yet: this is the motion model of the final filter and the first row of the ablation.

Causal: the pose at camera stamp t uses only state samples with stamp <= t.

    python3 tools_team/b0_dead_reckoning.py --state data/state_full.csv --gt data/gt_val.csv \
        --stamps data/images/index.csv --cam fl \
        --speed-col <vx column> --yawrate-col <yaw-rate column> [--vy-col <vy column>] \
        --out out/b0_val.csv
    python3 tools/evaluate.py data/gt_val.csv out/b0_val.csv

--gt is used ONLY for the initial pose (first row) and for the time window.
Column names: look at the header of state_full.csv (extract_state.py prints them).
"""
import argparse
import numpy as np
import pandas as pd


def integrate(ts, v, vy, r, x0, y0, yaw0, out_t):
    """Unicycle model with midpoint heading; returns poses at out_t (all >= ts[0])."""
    x, y, yaw = x0, y0, yaw0
    res, k = [], 0
    for i in range(len(ts) - 1):
        dt = (ts[i + 1] - ts[i]) / 1e9
        while k < len(out_t) and out_t[k] < ts[i + 1]:          # emit poses between samples
            h = (out_t[k] - ts[i]) / 1e9
            ym = yaw + 0.5 * r[i] * h
            res.append((out_t[k], x + (v[i]*np.cos(ym) - vy[i]*np.sin(ym)) * h,
                        y + (v[i]*np.sin(ym) + vy[i]*np.cos(ym)) * h, yaw + r[i] * h))
            k += 1
        ym = yaw + 0.5 * r[i] * dt
        x += (v[i] * np.cos(ym) - vy[i] * np.sin(ym)) * dt
        y += (v[i] * np.sin(ym) + vy[i] * np.cos(ym)) * dt
        yaw += r[i] * dt
    while k < len(out_t):                                        # after the last sample: hold
        res.append((out_t[k], x, y, yaw)); k += 1
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--state", required=True); ap.add_argument("--gt", required=True)
    ap.add_argument("--stamps", required=True); ap.add_argument("--cam", default="fl")
    ap.add_argument("--speed-col", required=True); ap.add_argument("--yawrate-col", required=True)
    ap.add_argument("--vy-col", default=None)
    ap.add_argument("--speed-scale", type=float, default=1.0)
    ap.add_argument("--yawrate-bias", type=float, default=0.0, help="rad/s, subtracted")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()

    gt = pd.read_csv(a.gt)
    t0, t1 = int(gt.timestamp_ns.iloc[0]), int(gt.timestamp_ns.iloc[-1])
    x0, y0, yaw0 = gt.x.iloc[0], gt.y.iloc[0], gt.yaw.iloc[0]          # allowed initial pose

    S = pd.read_csv(a.state).sort_values("timestamp_ns")
    S = S[(S.timestamp_ns >= t0) & (S.timestamp_ns <= t1)]
    ts = S.timestamp_ns.to_numpy(np.int64)
    v = S[a.speed_col].to_numpy(float) * a.speed_scale
    vy = S[a.vy_col].to_numpy(float) if a.vy_col else np.zeros(len(S))
    r = S[a.yawrate_col].to_numpy(float) - a.yawrate_bias

    idx = pd.read_csv(a.stamps)
    out_t = np.sort(idx[idx.cam == a.cam].stamp_ns.to_numpy(np.int64))
    out_t = out_t[(out_t >= ts[0]) & (out_t <= t1)]
    res = integrate(ts, v, vy, r, x0, y0, yaw0, out_t)
    df = pd.DataFrame(res, columns=["timestamp_ns", "x", "y", "yaw"])
    df["yaw"] = np.arctan2(np.sin(df.yaw), np.cos(df.yaw))
    df.to_csv(a.out, index=False, float_format="%.6f")
    print(f"{len(df)} poses ({(out_t[-1]-out_t[0])/1e9:.1f} s) -> {a.out}")


if __name__ == "__main__":
    main()
