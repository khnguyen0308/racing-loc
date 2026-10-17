#!/usr/bin/env python3
"""STEP 4b - camera timing: frame rate, dropped frames, offset of every camera to camera fl.
At 40 m/s a 10 ms offset is 0.4 m (INTRO slide 19), so this matters.

    python3 tools_team/check_sync.py data/images/index.csv
"""
import sys
import numpy as np
import pandas as pd


def main(path, ref="fl"):
    d = pd.read_csv(path)
    tr = np.sort(d[d.cam == ref].stamp_ns.to_numpy())
    print(f"{'cam':4s} {'frames':>7s} {'Hz':>6s} {'gaps>150ms':>10s} {'offset_to_'+ref+' median [ms]':>26s} {'p95 |offset| [ms]':>18s}")
    for cam, g in d.groupby("cam"):
        t = np.sort(g.stamp_ns.to_numpy())
        dt = np.diff(t) / 1e6
        j = np.clip(np.searchsorted(tr, t), 1, len(tr) - 1)
        nearest = np.where(np.abs(tr[j] - t) < np.abs(tr[j - 1] - t), tr[j], tr[j - 1])
        off = (t - nearest) / 1e6
        print(f"{cam:4s} {len(t):7d} {1000/np.median(dt):6.2f} {int((dt > 150).sum()):10d} {np.median(off):26.2f} {np.percentile(np.abs(off), 95):18.2f}")
    lag = (d.log_time_ns - d.stamp_ns) / 1e6
    print(f"\nlog_time - header.stamp (transport delay): median {lag.median():.1f} ms, p95 {lag.quantile(.95):.1f} ms")


if __name__ == "__main__":
    main(sys.argv[1])
