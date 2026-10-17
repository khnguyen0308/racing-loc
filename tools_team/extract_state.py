#!/usr/bin/env python3
"""STEP 3 - car state (/constructor0/state_estimation) -> CSV with EVERY numeric field.

!! The file also contains ground-truth pose / orientation / s, n. Those columns are only for
!! development & scoring. Only speed, accelerations, yaw rate (and similar motion fields) may be
!! used by the method. Decide the allowed column list as a team and write it in README.

    python3 tools_team/extract_state.py data/c0_..._camloc_0.mcap data/state_full.csv
"""
import csv, sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from tools_team.mcap_utils import flatten_numeric, iter_msgs, stamp_ns  # noqa: E402

TOPIC = "/constructor0/state_estimation"


def main(path, out):
    rows, cols = [], None
    for _, log_ns, msg in iter_msgs(path, [TOPIC]):
        d = flatten_numeric(msg)
        t = stamp_ns(msg.header) if hasattr(msg, "header") else log_ns
        d = {"timestamp_ns": t, "log_time_ns": log_ns, **{k: v for k, v in d.items() if not k.startswith("header.")}}
        if cols is None:
            cols = list(d)
        rows.append(d)
    with open(out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols, extrasaction="ignore")
        w.writeheader(); w.writerows(rows)
    print(f"{len(rows)} rows, {len(cols)} columns -> {out}")
    for c in cols:
        print("  ", c)


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
