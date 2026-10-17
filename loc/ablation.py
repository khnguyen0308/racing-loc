#!/usr/bin/env python3
"""Ablation on the validation laps 6-7: the same code and data, one component switched on per row.

    python loc/ablation.py                    # -> out/ablation/ablation_val.md (+ .csv), ~15 min (rows in parallel)

All rows: images at 10 Hz from the mcap, map + yaw-rate bias from laps 1-5 (never from 6-7).
This table is for REPORTING: every design choice was made on lap 5 before (map laps 1-4).
"""
import json, re, subprocess, sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
MCAP = ROOT / "data/c0_2026-10-13_22-53-42_camloc_0.mcap"
OUT = ROOT / "out/ablation"
ROWS = [  # name, extra args of loc/run_reloc.py
    ("B0: dead reckoning (speed + yaw rate)", ["--no-camera", "--no-vy", "--calib", OUT / "calib_zero.json"]),
    ("+ yaw-rate bias (laps 1-5)", ["--no-camera", "--no-vy"]),
    ("+ relocalization fr+fl (method 3a in EKF)", ["--cams", "fr", "fl", "--relock", "0", "--no-vy"]),
    ("fr+cr instead of fr+fl", ["--cams", "fr", "cr", "--relock", "0", "--no-vy"]),
    ("+ re-lock after consistent rejected fixes", ["--cams", "fr", "cr", "--no-vy"]),
    ("+ lateral speed vy in the prediction (final)", ["--cams", "fr", "cr"]),
]
KEYS = {"position RMSE": "pos_rmse", "lateral RMSE": "lat_rmse", "longitudinal RMSE": "lon_rmse",
        "yaw RMSE": "yaw_rmse_deg", "within 1.00 m": "within_1m_pct", "within 0.50 m": "within_0.5m_pct",
        "coverage": "coverage_pct"}


def score(gt, sub):
    txt = subprocess.run([sys.executable, "tools/evaluate.py", gt, sub], capture_output=True, text=True, cwd=ROOT, check=True).stdout
    out = {}
    for line in txt.splitlines():
        for k, name in KEYS.items():
            if line.startswith(k):
                out[name] = float(re.findall(r"[-+]?\d+\.\d+", line)[0])
        if line.startswith("position median/p95"):
            out["median"], out["p95"] = map(float, re.findall(r"\d+\.\d+", line)[:2])
    return out


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    json.dump({"yawrate_bias": 0.0}, open(OUT / "calib_zero.json", "w"))
    gt = str(ROOT / "data/gt_val.csv")
    base = [sys.executable, "loc/run_reloc.py", "--mcap", MCAP, "--gt", gt, "--map", ROOT / "data/map/laps1-5",
            "--calib", ROOT / "config/motion_calib_laps1-5.json"]
    procs = []
    for i, (name, extra) in enumerate(ROWS):
        out = OUT / f"row{i}.csv"
        log = open(OUT / f"row{i}.log", "w")
        procs.append((name, out, log, subprocess.Popen([str(c) for c in base + extra + ["--out", out]], cwd=ROOT,
                                                       stdout=log, stderr=subprocess.STDOUT)))
    rows = []
    for name, out, log, p in procs:
        p.wait(); log.close()
        if p.returncode:
            raise SystemExit(f"{name} failed, see {log.name}")
        rows.append(dict(config=name, **score(gt, str(out))))
        print(f"{name:48s} {rows[-1]['pos_rmse']:7.3f} m", flush=True)
    D = pd.DataFrame(rows)
    D.to_csv(OUT / "ablation_val.csv", index=False)
    md = ["| configuration (validation laps 6-7) | position RMSE | lateral | longitudinal | median / p95 | yaw RMSE | within 1 m |",
          "|---|---|---|---|---|---|---|"]
    for r in D.itertuples():
        md.append(f"| {r.config} | **{r.pos_rmse:.3f} m** | {r.lat_rmse:.3f} m | {r.lon_rmse:.3f} m | "
                  f"{r.median:.3f} / {r.p95:.3f} m | {r.yaw_rmse_deg:.3f}° | {r.within_1m_pct:.1f} % |")
    (OUT / "ablation_val.md").write_text("\n".join(md) + "\n", encoding="utf-8")
    print("\n".join(md))


if __name__ == "__main__":
    main()
