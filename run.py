#!/usr/bin/env python3
"""ONE COMMAND: raw mcap files -> pose CSV (timestamp_ns,x,y,yaw), one pose per image set at 10 Hz.

    # hidden test run (first pose given by the organisers as "timestamp_ns,x,y,yaw"),
    # map from the whole training run (all laps + pit lane)
    python run.py --final --test-mcap <test.mcap> --first-pose 1791918157114146050,7.1728,-9.9093,0.126255 \
                  --out out/submission.csv
    # reproduce our validation score (laps 6-7 of the training run, map from laps 1-5)
    python run.py --test-mcap data/c0_2026-10-13_22-53-42_camloc_0.mcap --first-pose-csv data/gt_val.csv \
                  --out out/submission_val.csv --eval-gt data/gt_val.csv

Stages (outputs are cached in data/ and config/; --rebuild recomputes them):
  training run  1 GT + static + car state      tools/extract_gt.py, tools_team/extract_static.py, extract_state.py
                2 images 5 Hz for the map      tools_team/extract_images.py
                3 laps, car-body masks         tools_team/track_and_laps.py, tools_team/masks.py
                4 yaw-rate bias, 3D map        loc/calib_motion.py, loc/build_map.py      (GT used: training only)
  test run      5 car state (allowed columns)  tools_team/extract_state.py + state_io.load_state
                6 causal EKF + relocalization  loc/run_reloc.py (images decoded from the mcap at 10 Hz)
Ground truth of the test run is never read; --first-pose(-csv) gives only the first pose
(and, with a multi-row CSV such as data/gt_val.csv, the end of the time window for validation).
"""
import argparse, json, subprocess, sys, time
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent
sys.path[:0] = [str(ROOT), str(ROOT / "tools_team")]
from loc.vision import CAMS  # noqa: E402
PY = sys.executable


def step(name, out, cmd, rebuild):
    """Run cmd unless every path in `out` exists (cache)."""
    outs = [Path(o) for o in (out if isinstance(out, list) else [out])]
    if not rebuild and all(o.exists() for o in outs):
        print(f"[skip] {name}: {', '.join(str(o.relative_to(ROOT)) for o in outs)} exists")
        return
    print(f"[run ] {name}", flush=True)
    tic = time.time()
    subprocess.run([PY] + [str(c) for c in cmd], check=True, cwd=ROOT)
    print(f"[done] {name} ({time.time() - tic:.0f} s)", flush=True)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--test-mcap", required=True, help="mcap of the run to localize")
    ap.add_argument("--first-pose", help="timestamp_ns,x,y,yaw of the first pose of the run")
    ap.add_argument("--first-pose-csv", help="CSV timestamp_ns,x,y,yaw: row 1 = first pose; last row = end of window")
    ap.add_argument("--end-ns", type=int, help="end of the time window (default: end of the CSV or of the mcap)")
    ap.add_argument("--out", required=True)
    ap.add_argument("--train-mcap", default=str(ROOT / "data/c0_2026-10-13_22-53-42_camloc_0.mcap"))
    ap.add_argument("--map-laps", type=int, nargs="+", default=[1, 2, 3, 4, 5],
                    help="training laps for the map + bias (1-5 keeps laps 6-7 for validation; 1-7 for the final test run)")
    ap.add_argument("--final", action="store_true",
                    help="hidden test run: map from the WHOLE training run (all 7 laps + pit lane / out-laps), bias on laps 1-7")
    ap.add_argument("--cams", nargs="+", default=list(CAMS))
    ap.add_argument("--eval-gt", help="score the output with tools/evaluate.py against this GT (development only)")
    ap.add_argument("--rebuild", action="store_true")
    a = ap.parse_args()
    D, R = ROOT / "data", a.rebuild
    if a.final:
        a.map_laps = list(range(1, 8))
    tag = "whole" if a.final else (f"laps{a.map_laps[0]}-{a.map_laps[-1]}"
                                   if a.map_laps == list(range(a.map_laps[0], a.map_laps[-1] + 1))
                                   else "laps" + "_".join(map(str, a.map_laps)))
    allow = ["--allow-val-laps"] if set(a.map_laps) & {6, 7} else []
    if allow:
        print("NOTE: map includes laps 6-7 -> use only for the hidden test run, a validation score would be unfair")

    # ---- training run -> map ------------------------------------------------------------------------------
    step("GT (training run)", D / "gt.csv", ["tools/extract_gt.py", a.train_mcap, D / "gt.csv"], R)
    step("calibration + extrinsics + borders", D / "static.json", ["tools_team/extract_static.py", a.train_mcap, D / "static.json"], R)
    step("car state (training run)", D / "state_full.csv", ["tools_team/extract_state.py", a.train_mcap, D / "state_full.csv"], R)
    step("map images 5 Hz", [D / "map_images" / "index.csv"] + [D / "map_images" / c for c in a.cams],
         ["tools_team/extract_images.py", a.train_mcap, D / "map_images", "--cams", *a.cams, "--save-every", 2], R)
    (ROOT / "out").mkdir(exist_ok=True)
    step("laps", D / "laps.csv", ["tools_team/track_and_laps.py", D / "static.json", D / "gt.csv", D / "laps.csv",
                                  ROOT / "out/track_gt.png"], R)
    step("car-body masks", [D / "masks" / f"{c}_valid.png" for c in a.cams],
         ["tools_team/masks.py", "--images", D / "map_images", "--out", D / "masks", "--auto", "--state", D / "state_full.csv"], R)
    calib = ROOT / f"config/motion_calib_{tag}.json"
    step("yaw-rate bias", calib, ["loc/calib_motion.py", "--laps", *a.map_laps, "--out", calib, *allow], R)
    mp = D / "map" / tag
    step("3D map", [Path(f"{mp}_{c}.npz") for c in a.cams],
         ["loc/build_map.py", "--laps", *a.map_laps, "--cams", *a.cams, "--out", mp, *allow,
          *(["--whole-run"] if a.final else [])], R)

    # ---- test run -> poses --------------------------------------------------------------------------------
    work = ROOT / "out" / ("run_" + Path(a.test_mcap).stem)
    work.mkdir(parents=True, exist_ok=True)
    test_state = D / "state_full.csv" if Path(a.test_mcap).resolve() == Path(a.train_mcap).resolve() else work / "state.csv"
    step("car state (test run)", test_state, ["tools_team/extract_state.py", a.test_mcap, test_state], R)

    if a.first_pose:
        t0, x0, y0, yaw0 = (float(v) for v in a.first_pose.split(",")); t0 = int(t0); t_end = None
    elif a.first_pose_csv:
        fp = pd.read_csv(a.first_pose_csv)
        t0, x0, y0, yaw0 = int(fp.timestamp_ns.iloc[0]), fp.x.iloc[0], fp.y.iloc[0], fp.yaw.iloc[0]
        t_end = int(fp.timestamp_ns.iloc[-1]) if len(fp) > 1 else None
    else:
        raise SystemExit("give --first-pose or --first-pose-csv")
    t1 = a.end_ns or t_end or np.iinfo(np.int64).max // 2

    from loc.reloc import Relocalizer
    from loc.run_reloc import localize, mcap_source, summary
    from state_io import load_state
    S = load_state(test_state, ["v_mps", "wz_radps", "vy_mps"])
    rl = Relocalizer(json.load(open(D / "static.json")), str(mp), cams=a.cams)
    print(f"[run ] localize {Path(a.test_mcap).name} from t0={t0} with map {tag} ({'+'.join(a.cams)})", flush=True)
    tic = time.time()
    P, G, ms, f = localize(mcap_source(a.test_mcap, a.cams, t0, t1), S, (x0, y0, yaw0), t0, rl,
                           json.load(open(calib))["yawrate_bias"], verbose_every=1000)
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    P[["timestamp_ns", "x", "y", "yaw"]].to_csv(a.out, index=False, float_format="%.6f")
    G.to_csv(work / "reloc_log.csv", index=False)
    print(f"[done] {summary(P, G, ms, f)} | wall {time.time() - tic:.0f} s -> {a.out}")
    if a.eval_gt:
        subprocess.run([PY, "tools/evaluate.py", a.eval_gt, a.out], check=True, cwd=ROOT)


if __name__ == "__main__":
    main()
