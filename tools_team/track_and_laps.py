#!/usr/bin/env python3
"""STEP 5 - track geometry + lap split + plot (README step 2/3 of the INTRO).

 * centre line of the race track = midpoints of track_left / track_right
 * start/finish gate = segment from track_left[0] to the closest track_right point
 * a lap = GT path between two forward crossings of the gate
 * frames close to the pit-lane borders are flagged as "pit"

    python3 tools_team/track_and_laps.py data/static.json data/gt.csv data/laps.csv data/track_gt.png
"""
import json, sys
import numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.spatial import cKDTree


def load_gt(path):
    d = np.genfromtxt(path, delimiter=",", names=True, dtype=None, encoding=None)
    return d["timestamp_ns"].astype(np.int64), np.stack([d["x"], d["y"]], 1).astype(float), d["yaw"].astype(float)


def resample(poly, step=1.0):
    seg = np.linalg.norm(np.diff(poly, axis=0), axis=1)
    s = np.concatenate([[0], np.cumsum(seg)])
    q = np.arange(0, s[-1], step)
    return np.stack([np.interp(q, s, poly[:, 0]), np.interp(q, s, poly[:, 1])], 1)


def centre_line(left, right, step=1.0):
    L = resample(left, step)
    _, j = cKDTree(right).query(L)
    C = (L + right[j]) / 2
    width = np.linalg.norm(L - right[j], axis=1)
    return resample(C, step), width


def seg_cross(p0, p1, a, b):
    """True if segment p0-p1 crosses segment a-b."""
    def orient(u, v, w):
        return np.sign((v[0]-u[0])*(w[1]-u[1]) - (v[1]-u[1])*(w[0]-u[0]))
    return orient(a, b, p0) != orient(a, b, p1) and orient(p0, p1, a) != orient(p0, p1, b)


def main(static_path, gt_path, out_csv, out_png):
    st = json.load(open(static_path))
    B = {k: np.array(v["points"])[:, :2] for k, v in st["track_bounds"].items()}
    left, right = B["track_left"], B["track_right"]
    C, width = centre_line(left, right)
    t, xy, yaw = load_gt(gt_path)
    lap_len = np.sum(np.linalg.norm(np.diff(C, axis=0), axis=1))
    print(f"centre line {lap_len:.0f} m, track width median {np.median(width):.1f} m (min {width.min():.1f}, max {width.max():.1f})")

    a = left[0]; b = right[np.argmin(np.linalg.norm(right - a, axis=1))]
    fwd = C[1] - C[0] if np.linalg.norm(C[0] - (a + b) / 2) < np.linalg.norm(C[-1] - (a + b) / 2) else C[-1] - C[-2]
    cross = []
    for i in range(1, len(xy)):
        if seg_cross(xy[i-1], xy[i], a, b) and np.dot(xy[i] - xy[i-1], fwd) > 0:
            if not cross or (t[i] - t[cross[-1]]) / 1e9 > 20:          # ignore double hits < 20 s apart
                cross.append(i)

    pit = np.zeros(len(xy), bool)
    if "pitlane_left" in B and "pitlane_right" in B:
        d_pit = np.minimum(cKDTree(B["pitlane_left"]).query(xy)[0], cKDTree(B["pitlane_right"]).query(xy)[0])
        d_trk = cKDTree(C).query(xy)[0]
        pit = (d_pit < 8) & (d_trk > np.median(width) / 2 + 1)

    speed = np.r_[0, np.linalg.norm(np.diff(xy, axis=0), axis=1) / np.maximum(np.diff(t) / 1e9, 1e-6)]
    rows = []
    for k in range(len(cross) - 1):
        i0, i1 = cross[k], cross[k + 1]
        rows.append((k + 1, t[i0], t[i1], (t[i1] - t[i0]) / 1e9, np.percentile(speed[i0:i1], 99), pit[i0:i1].mean() * 100))
    with open(out_csv, "w") as f:
        f.write("lap,start_ns,end_ns,duration_s,v_max_mps,pit_pct\n")
        for r in rows:
            f.write(f"{r[0]},{r[1]},{r[2]},{r[3]:.2f},{r[4]:.1f},{r[5]:.1f}\n")
            print(f"lap {r[0]:2d}: {r[3]:7.1f} s  v_max {r[4]:5.1f} m/s  pit {r[5]:4.1f} %")
    print(f"{len(cross)} gate crossings -> {len(rows)} full laps -> {out_csv}")

    fig, ax = plt.subplots(figsize=(10, 10))
    for k, p in B.items():
        ax.plot(p[:, 0], p[:, 1], lw=0.8, color="0.3" if "track" in k else "tab:orange", label=k)
    ax.plot(C[:, 0], C[:, 1], "--", lw=0.6, color="0.6", label="centre (computed)")
    sc = ax.scatter(xy[::10, 0], xy[::10, 1], c=speed[::10], s=1, cmap="viridis")
    ax.plot([a[0], b[0]], [a[1], b[1]], "r-", lw=2, label="start gate")
    ax.set_aspect("equal"); ax.legend(loc="best", fontsize=7); fig.colorbar(sc, label="GT speed [m/s]")
    fig.savefig(out_png, dpi=150); print(f"-> {out_png}")
    np.save(out_csv.replace(".csv", "_centre.npy"), C)


if __name__ == "__main__":
    main(*sys.argv[1:5])