#!/usr/bin/env python3
"""Causal EKF run: car state (allowed columns only) + optional camera measurements -> CSV at camera rate.

    python3 loc/run_ekf.py --state data/state_full.csv --gt data/gt_val.csv --stamps data/images/index.csv \
        --out out/v1_val.csv [--meas out/border_meas.csv]

--gt is used ONLY for the first pose (row 1) and the time window.
--meas (optional): CSV timestamp_ns,x,y,yaw,sig_lon,sig_lat,sig_yaw  (camera measurements, causal)
"""
import argparse, sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "tools_team")]
from loc.ekf import CarEKF, body_cov, wrap  # noqa: E402
from state_io import load_state            # noqa: E402


def run(state, t_out, first_pose, t_first, meas=None, ekf_kw=None, gate=11.34, use_vy=False):
    """first_pose = GT pose at time t_first (ns); integration starts there."""
    ts = state.timestamp_ns.to_numpy(np.int64)
    v, wz = state.v_mps.to_numpy(float), state.wz_radps.to_numpy(float)
    vy = state.vy_mps.to_numpy(float) if use_vy else np.zeros(len(state))
    f = CarEKF(*first_pose, **(ekf_kw or {}))
    mi, mt = 0, (meas.timestamp_ns.to_numpy(np.int64) if meas is not None else np.array([], np.int64))
    out, t_cur = [], int(t_first)
    j = max(np.searchsorted(ts, t_cur, side="right") - 1, 0)   # last state sample <= t
    for t in t_out:
        while j + 1 < len(ts) and ts[j + 1] <= t:              # integrate state samples up to t
            f.predict(v[j], wz[j], (ts[j + 1] - t_cur) / 1e9, vy[j]); t_cur = ts[j + 1]; j += 1
        f.predict(v[j], wz[j], (t - t_cur) / 1e9, vy[j]); t_cur = t  # hold last sample until t (causal)
        while mi < len(mt) and mt[mi] <= t:                    # camera measurements with stamp <= t
            m = meas.iloc[mi]
            f.update_pose((m.x, m.y, m.yaw), body_cov(m.yaw, m.sig_lon, m.sig_lat, m.sig_yaw), gate)
            mi += 1
        out.append((t, f.X[0], f.X[1], wrap(f.X[2]), f.X[3], f.X[4], np.sqrt(f.P[0, 0] + f.P[1, 1])))
    return pd.DataFrame(out, columns=["timestamp_ns", "x", "y", "yaw", "bias_w", "k_v", "sigma_pos"]), f


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--state", required=True); ap.add_argument("--gt", required=True)
    ap.add_argument("--stamps", required=True); ap.add_argument("--cam", default="fl")
    ap.add_argument("--meas", default=None); ap.add_argument("--out", required=True)
    ap.add_argument("--use-vy", action="store_true")
    a = ap.parse_args()
    gt = pd.read_csv(a.gt)
    t0, t1 = int(gt.timestamp_ns.iloc[0]), int(gt.timestamp_ns.iloc[-1])
    S = load_state(a.state, ["v_mps", "wz_radps", "vy_mps"])
    idx = pd.read_csv(a.stamps)
    t_out = np.sort(idx[idx.cam == a.cam].stamp_ns.to_numpy(np.int64))
    t_out = t_out[(t_out >= t0) & (t_out <= t1)]
    meas = pd.read_csv(a.meas).sort_values("timestamp_ns") if a.meas else None
    df, f = run(S, t_out, (gt.x.iloc[0], gt.y.iloc[0], gt.yaw.iloc[0]), t0, meas, use_vy=a.use_vy)
    df[["timestamp_ns", "x", "y", "yaw"]].to_csv(a.out, index=False, float_format="%.6f")
    df.to_csv(a.out.replace(".csv", "_debug.csv"), index=False)
    print(f"{len(df)} poses -> {a.out} | updates {f.n_upd}, rejected {f.n_rej} | "
          f"final bias {f.X[3]:+.6f} rad/s, k_v {f.X[4]:.5f}")


if __name__ == "__main__":
    main()
