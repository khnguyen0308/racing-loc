#!/usr/bin/env python3
"""STEP 6 - validation ground truth for chosen laps (INTRO: keep 2 laps, never train/tune on them).

    python3 tools_team/crop_gt.py data/gt.csv data/laps.csv 7 8 --out data/gt_val.csv
Writes gt_val.csv (same columns as gt.csv) and prints the first pose (= allowed initial pose).
Score a submission on the validation laps with:  python3 tools/evaluate.py data/gt_val.csv sub_val.csv
"""
import argparse
import numpy as np


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("gt"); ap.add_argument("laps"); ap.add_argument("ids", type=int, nargs="+")
    ap.add_argument("--out", default="data/gt_val.csv")
    a = ap.parse_args()
    L = np.genfromtxt(a.laps, delimiter=",", names=True, dtype=None, encoding=None)
    sel = L[np.isin(L["lap"], a.ids)]
    t0, t1 = int(sel["start_ns"].min()), int(sel["end_ns"].max())
    lines = open(a.gt).read().splitlines()
    keep = [lines[0]] + [ln for ln in lines[1:] if t0 <= int(ln.split(",")[0]) <= t1]
    open(a.out, "w").write("\n".join(keep) + "\n")
    print(f"laps {a.ids}: {(t1 - t0) / 1e9:.1f} s, {len(keep) - 1} GT poses -> {a.out}")
    print("initial pose (allowed):", keep[1])


if __name__ == "__main__":
    main()
