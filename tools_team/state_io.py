"""Load ONLY the allowed car-state columns (guards against using ground truth by accident).

    from state_io import load_state
    S = load_state("data/state_full.csv")     # pandas DataFrame, timestamp_ns + allowed columns
"""
import fnmatch
from pathlib import Path

import pandas as pd

CFG = Path(__file__).resolve().parents[1] / "config" / "state_columns.yaml"


def _lists():
    import yaml
    c = yaml.safe_load(open(CFG))
    allowed = [x for v in c["allowed"].values() for x in v]
    return allowed, c["forbidden"]


def load_state(path, columns=None):
    allowed, forbidden = _lists()
    want = columns or allowed
    for col in want:
        if any(fnmatch.fnmatch(col, f) for f in forbidden) or col not in allowed:
            raise ValueError(f"column '{col}' is not allowed at inference (see config/state_columns.yaml)")
    head = pd.read_csv(path, nrows=0).columns
    use = ["timestamp_ns"] + [c for c in want if c in head]
    return pd.read_csv(path, usecols=use).sort_values("timestamp_ns").reset_index(drop=True)
