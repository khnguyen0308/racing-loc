#!/usr/bin/env python3
"""PLANNING TOOL ONLY - never for a submission (it reads the GROUND TRUTH).

Simulates camera measurements = GT + noise (in the car frame) to answer:
"how good / how frequent must our border-matching measurement be to reach score X?"

    python3 loc/whatif_measurements.py --state data/state_full.csv --gt data/gt_val.csv --stamps data/images/index.csv
"""
import argparse, sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "tools_team")]
from loc.run_ekf import run      # noqa: E402
from state_io import load_state  # noqa: E402


def score(gt, sub):
    t = gt.timestamp_ns.to_numpy(np.int64); ts = sub.timestamp_ns.to_numpy(np.int64)
    gx = np.interp(ts, t, gt.x); gy = np.interp(ts, t, gt.y); gyaw = np.interp(ts, t, np.unwrap(gt.yaw))
    dx, dy = sub.x - gx, sub.y - gy
    lat = -np.sin(gyaw) * dx + np.cos(gyaw) * dy
    pos = np.hypot(dx, dy); dyaw = np.degrees(np.angle(np.exp(1j * (sub.yaw - gyaw))))
    r = lambda v: float(np.sqrt(np.mean(np.square(v))))
    return dict(pos=r(pos), lat=r(lat), yaw=r(dyaw), within1=100 * float(np.mean(pos < 1)))


def simulate(gt, t_cam, rng, rate_hz=10, sig_lat=0.3, sig_yaw_deg=0.5, sig_lon_corner=None,
             kappa_corner=0.01, p_out=0.0, out_mag=8.0, p_drop=0.0):
    t = gt.timestamp_ns.to_numpy(np.int64)
    yaw_u = np.unwrap(gt.yaw.to_numpy())
    every = max(1, int(round(10 / rate_hz)))
    tm = t_cam[::every]
    x = np.interp(tm, t, gt.x); y = np.interp(tm, t, gt.y); yw = np.interp(tm, t, yaw_u)
    # curvature of the GT path -> "corner" where the borders also constrain the along-track position
    tt = t / 1e9; dyaw = np.gradient(yaw_u, tt); v = np.hypot(np.gradient(gt.x, tt), np.gradient(gt.y, tt))
    kappa = np.interp(tm, t, np.abs(dyaw) / np.maximum(v, 1.0))
    n = len(tm)
    sig_lon = np.where((sig_lon_corner is not None) & (kappa > kappa_corner), sig_lon_corner or 1.0, 1e3)
    e_lon = rng.normal(0, np.minimum(sig_lon, 50.0)); e_lat = rng.normal(0, sig_lat, n)
    e_yaw = rng.normal(0, np.radians(sig_yaw_deg), n)
    out = rng.random(n) < p_out                                    # gross outliers (wrong border match)
    e_lat[out] += rng.choice([-1, 1], out.sum()) * rng.uniform(0.5, 1.5, out.sum()) * out_mag
    c, s = np.cos(yw), np.sin(yw)
    df = pd.DataFrame(dict(timestamp_ns=tm, x=x + c * e_lon - s * e_lat, y=y + s * e_lon + c * e_lat,
                           yaw=yw + e_yaw, sig_lon=sig_lon, sig_lat=sig_lat, sig_yaw=np.radians(sig_yaw_deg)))
    keep = rng.random(n) >= p_drop
    return df[keep].reset_index(drop=True), float(np.mean(kappa > kappa_corner))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--state", required=True); ap.add_argument("--gt", required=True)
    ap.add_argument("--stamps", required=True); ap.add_argument("--cam", default="fl")
    ap.add_argument("--seeds", type=int, default=3)
    a = ap.parse_args()
    gt = pd.read_csv(a.gt)
    t0, t1 = int(gt.timestamp_ns.iloc[0]), int(gt.timestamp_ns.iloc[-1])
    S = load_state(a.state, ["v_mps", "wz_radps", "vy_mps"])
    idx = pd.read_csv(a.stamps)
    t_cam = np.sort(idx[idx.cam == a.cam].stamp_ns.to_numpy(np.int64)); t_cam = t_cam[(t_cam >= t0) & (t_cam <= t1)]
    first = (gt.x.iloc[0], gt.y.iloc[0], gt.yaw.iloc[0])
    scen = [
        ("no camera (= B0)", None),
        ("lat 0.3m + yaw 0.5deg, 10 Hz, NO along-track info", dict(sig_lat=0.3)),
        ("+ along-track 1.0m in corners", dict(sig_lat=0.3, sig_lon_corner=1.0)),
        ("same, only 2 Hz", dict(sig_lat=0.3, sig_lon_corner=1.0, rate_hz=2)),
        ("same 10 Hz, 50% frames missing", dict(sig_lat=0.3, sig_lon_corner=1.0, p_drop=0.5)),
        ("lat 0.6m (worse detector)", dict(sig_lat=0.6, sig_lon_corner=1.0)),
        ("lat 0.15m (excellent detector)", dict(sig_lat=0.15, sig_lon_corner=0.5)),
        ("10% outliers 4-12 m, gating ON", dict(sig_lat=0.3, sig_lon_corner=1.0, p_out=0.1)),
        ("10% outliers 4-12 m, gating OFF", dict(sig_lat=0.3, sig_lon_corner=1.0, p_out=0.1, _nogate=True)),
    ]
    print(f"validation window {(t1-t0)/1e9:.1f} s, {len(t_cam)} camera frames, seeds={a.seeds}")
    print(f"{'scenario':52s} {'pos RMSE':>9s} {'lat RMSE':>9s} {'yaw RMSE':>9s} {'<1m %':>7s} {'rejected':>9s}")
    for name, kw in scen:
        res, rej = [], []
        for sd in range(a.seeds if kw else 1):
            rng = np.random.default_rng(sd)
            meas = None; gate = 11.34
            if kw:
                kw2 = dict(kw); nogate = kw2.pop("_nogate", False); gate = None if nogate else 11.34
                meas, corner_frac = simulate(gt, t_cam, rng, **kw2)
            df, f = run(S, t_cam, first, t0, meas, gate=gate)
            res.append(score(gt, df)); rej.append(f.n_rej / max(f.n_upd + f.n_rej, 1))
        m = {k: np.mean([r[k] for r in res]) for k in res[0]}
        print(f"{name:52s} {m['pos']:9.3f} {m['lat']:9.3f} {m['yaw']:9.3f} {m['within1']:7.1f} {100*np.mean(rej):8.1f}%")
    print(f"(share of camera frames counted as 'corner', |kappa|>0.01 1/m: {100*corner_frac:.0f} %)")


if __name__ == "__main__":
    main()
