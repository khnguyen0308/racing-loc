#!/usr/bin/env python3
"""Figures for the report / slides (development only: reads GT to show errors).

    python loc/plots.py --sub out/submission_val.csv --gt data/gt_val.csv \
        --log out/run_c0_2026-10-13_22-53-42_camloc_0/reloc_log.csv --ablation out/ablation/ablation_val.csv

-> out/figures/{trajectory,error_time,error_map,ablation,pipeline}.png
"""
import argparse, json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.collections import LineCollection
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.patches import FancyBboxPatch, Rectangle

ROOT = Path(__file__).resolve().parents[1]
# reference palette (dataviz skill, light mode)
SURF, INK, INK2, MUTED, GRID, AXIS = "#fcfcfb", "#0b0b0b", "#52514e", "#898781", "#e1e0d9", "#c3c2b7"
S1, S2 = "#2a78d6", "#eb6834"                                      # categorical slots 1, 2
SEQ = LinearSegmentedColormap.from_list("blue_seq", ["#cde2fb", "#86b6ef", "#3987e5", "#256abf", "#184f95", "#0d366b"])
plt.rcParams.update({"figure.facecolor": SURF, "axes.facecolor": SURF, "savefig.facecolor": SURF,
                     "axes.edgecolor": AXIS, "axes.labelcolor": INK2, "xtick.color": MUTED, "ytick.color": MUTED,
                     "text.color": INK, "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.6,
                     "axes.spines.top": False, "axes.spines.right": False, "font.size": 10,
                     "font.family": ["Segoe UI", "DejaVu Sans"], "legend.frameon": False})


def errors(sub, gt):
    t = sub.timestamp_ns.to_numpy(np.int64); tg = gt.timestamp_ns.to_numpy(np.int64)
    gx, gy = np.interp(t, tg, gt.x), np.interp(t, tg, gt.y); gyaw = np.interp(t, tg, np.unwrap(gt.yaw))
    dx, dy = sub.x.to_numpy() - gx, sub.y.to_numpy() - gy
    return dict(t=(t - t[0]) / 1e9, gx=gx, gy=gy, pos=np.hypot(dx, dy),
                lat=-np.sin(gyaw) * dx + np.cos(gyaw) * dy, lon=np.cos(gyaw) * dx + np.sin(gyaw) * dy)


def borders(ax, static, lw=0.7):
    for k, v in static["track_bounds"].items():
        p = np.array(v["points"])
        ax.plot(p[:, 0], p[:, 1], color=AXIS if "track" in k else GRID, lw=lw, zorder=1)


def fig_trajectory(sub, gt, E, static, out):
    fig, (a, b) = plt.subplots(1, 2, figsize=(13, 6.2), gridspec_kw=dict(width_ratios=[1.35, 1]))
    for ax in (a, b):
        borders(ax, static)
        ax.plot(gt.x, gt.y, color=INK, lw=2.0, label="ground truth", zorder=2)
        ax.plot(sub.x, sub.y, color=S1, lw=1.4, ls=(0, (4, 2)), label="estimate (camera + EKF)", zorder=3)
        ax.set_aspect("equal"); ax.grid(False); ax.set_xlabel("x [m]"); ax.set_ylabel("y [m]")
    k = int(np.argmax(E["pos"])); cx, cy = E["gx"][k], E["gy"][k]
    b.set_xlim(cx - 30, cx + 30); b.set_ylim(cy - 30, cy + 30)
    a.add_patch(Rectangle((cx - 30, cy - 30), 60, 60, fill=False, ec=S2, lw=1.5, zorder=4))
    a.set_title("Validation laps 6-7: estimate on top of ground truth", loc="left", fontsize=11)
    b.set_title(f"Zoom 60 m around the largest error ({E['pos'][k]:.2f} m)", loc="left", fontsize=11)
    a.legend(loc="lower right")                                     # below the last straight: empty corner
    fig.tight_layout(); fig.savefig(out / "trajectory.png", dpi=170); plt.close(fig)


def fig_error_time(E, log, out):
    fig, (a, r) = plt.subplots(2, 1, figsize=(12, 5.2), sharex=True, gridspec_kw=dict(height_ratios=[4, 1]))
    a.plot(E["t"], E["pos"], color=S1, lw=1.5, label="position error")
    a.plot(E["t"], np.abs(E["lat"]), color=S2, lw=1.0, label="|lateral| error")
    a.axhline(1.0, color=MUTED, lw=1, ls="--")
    a.text(E["t"][-1], 1.02, "target < 1 m", ha="right", va="bottom", color=INK2, fontsize=9)
    a.set_ylim(0, max(1.15, E["pos"].max() * 1.1)); a.set_ylabel("error [m]")
    rm = np.sqrt(np.mean(E["pos"] ** 2))
    a.set_title(f"Error over time, validation laps 6-7 (position RMSE {rm:.3f} m)", loc="left", fontsize=11)
    a.legend(loc="upper left", ncol=2)
    t0 = None
    if log is not None and len(log):
        t0 = log.timestamp_ns.min()
    for i, (cam, col) in enumerate([("fr", S1), ("cr", S2)]):
        if log is None:
            break
        u = log[(log.cam == cam) & log.used]
        r.vlines((u.timestamp_ns - log.timestamp_ns.min()) / 1e9 + (E["t"][0]), i + 0.15, i + 0.85, color=col, lw=0.4)
        r.text(-2, i + 0.5, cam, ha="right", va="center", color=INK2, fontsize=9)
    r.set_yticks([]); r.set_ylim(0, 2); r.grid(False); r.set_xlabel("time since first pose [s]")
    r.set_title("accepted camera fixes", loc="left", fontsize=9, color=INK2)
    fig.tight_layout(); fig.savefig(out / "error_time.png", dpi=170); plt.close(fig)


def fig_error_map(E, static, out):
    fig, ax = plt.subplots(figsize=(8.5, 7.5))
    borders(ax, static, lw=0.5)
    pts = np.c_[E["gx"], E["gy"]]
    seg = np.stack([pts[:-1], pts[1:]], 1)
    vmax = max(0.5, float(np.percentile(E["pos"], 99)))
    lc = LineCollection(seg, cmap=SEQ, norm=plt.Normalize(0, vmax), lw=4, capstyle="round", zorder=3)
    lc.set_array(E["pos"][:-1]); ax.add_collection(lc)
    cb = fig.colorbar(lc, ax=ax, shrink=0.7, pad=0.02); cb.set_label("position error [m]"); cb.outline.set_visible(False)
    ax.set_aspect("equal"); ax.grid(False); ax.set_xlabel("x [m]"); ax.set_ylabel("y [m]")
    ax.set_title("Where on the track the error is (validation laps 6-7)", loc="left", fontsize=11)
    fig.tight_layout(); fig.savefig(out / "error_map.png", dpi=170); plt.close(fig)


def fig_ablation(A, out):
    fig, ax = plt.subplots(figsize=(10, 0.6 * len(A) + 1.4))
    y = np.arange(len(A))[::-1]
    ax.hlines(y, A.pos_rmse.min() * 0.6, A.pos_rmse, color=GRID, lw=1)
    ax.plot(A.pos_rmse, y, "o", color=S1, ms=9, mec=SURF, mew=2)
    for yi, v in zip(y, A.pos_rmse):
        ax.text(v * 1.12, yi, f"{v:.3f} m", va="center", color=INK, fontsize=10)
    ax.set_yticks(y); ax.set_yticklabels(A.config, color=INK2)
    ax.set_xscale("log"); ax.set_xlim(A.pos_rmse.min() * 0.6, A.pos_rmse.max() * 2.2)
    ax.axvline(1.0, color=MUTED, lw=1, ls="--"); ax.text(1.06, y.max() + 0.6, "1 m target", ha="left", color=INK2, fontsize=9)
    ax.set_xlabel("position RMSE [m], log scale (lower is better)"); ax.grid(axis="y", visible=False)
    ax.set_title("Ablation on validation laps 6-7: one component added per row", loc="left", fontsize=11)
    ax.set_ylim(-0.6, y.max() + 1)
    fig.tight_layout(); fig.savefig(out / "ablation.png", dpi=170); plt.close(fig)


def fig_pipeline(out):
    fig, ax = plt.subplots(figsize=(13, 5.6)); ax.set_xlim(0, 13); ax.set_ylim(0, 5.6); ax.axis("off")

    def box(x, y, w, h, title, body, ec=AXIS):
        ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.02,rounding_size=0.12", fc=SURF, ec=ec, lw=1.4))
        ax.text(x + 0.15, y + h - 0.22, title, fontsize=10.5, weight="bold", va="top", color=INK)
        ax.text(x + 0.15, y + h - 0.58, body, fontsize=8.8, va="top", color=INK2, linespacing=1.35)

    def arrow(x0, y0, x1, y1):
        ax.annotate("", (x1, y1), (x0, y0), arrowprops=dict(arrowstyle="-|>", color=MUTED, lw=1.3))
    # each offline block sits right above the online block it feeds: all inter-row arrows are vertical
    ax.text(0.1, 5.4, "A  Offline, once — training run (ground truth allowed)", fontsize=11, weight="bold", color=S2)
    box(0.1, 3.1, 2.6, 1.6, "Training mcap", "images fr, cr @ 5 Hz\nGT pose (x, y, z, yaw, pitch, roll)\ncar state, calibration")
    box(3.0, 3.1, 2.6, 1.6, "Motion calibration", "yaw-rate bias\n(training laps only)")
    box(5.9, 3.1, 3.3, 1.6, "3D feature map", "SIFT on CLAHE image, body masked\nmatch keyframe pairs, epipolar\ncheck with GT pose, triangulate\n-> ~1 M 3D points + descriptors")
    arrow(2.7, 3.9, 3.0, 3.9)
    ax.plot([1.4, 1.4, 7.55], [4.7, 4.95, 4.95], color=MUTED, lw=1.3)        # elbow over the calibration box
    arrow(7.55, 4.95, 7.55, 4.7)
    ax.text(0.1, 2.5, "B  Online, causal, 10 Hz", fontsize=11, weight="bold", color=S1)
    ax.text(9.5, 2.5, "run to localize: no ground truth, first pose only", fontsize=9, color=INK2)
    box(0.1, 0.5, 2.6, 1.7, "Car state  (t' <= t)", "speed v, yaw rate wz,\nlateral speed vy")
    box(3.0, 0.5, 2.6, 1.7, "EKF predict", "x, y, yaw, yaw-rate bias,\nspeed scale\nunicycle + sideslip model")
    box(5.9, 0.5, 3.3, 1.7, "Relocalize  (images fr, cr at t)", "keyframes near the predicted pose\n-> SIFT match -> PnP RANSAC\n-> car pose; check: car on the\nroad (map height)")
    box(9.5, 0.5, 3.4, 1.7, "EKF update -> pose CSV", "chi2 gate rejects bad fixes\nre-lock when 5 rejected fixes\nagree (filter was wrong)\nwrite timestamp_ns, x, y, yaw", ec=S1)
    arrow(2.7, 1.35, 3.0, 1.35); arrow(5.6, 1.35, 5.9, 1.35); arrow(9.2, 1.35, 9.5, 1.35)
    arrow(4.3, 3.1, 4.3, 2.2); arrow(7.55, 3.1, 7.55, 2.2)
    fig.savefig(out / "pipeline.png", dpi=170, bbox_inches="tight"); plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sub", default=str(ROOT / "out/submission_val.csv")); ap.add_argument("--gt", default=str(ROOT / "data/gt_val.csv"))
    ap.add_argument("--log", default=None); ap.add_argument("--ablation", default=None)
    ap.add_argument("--out", default=str(ROOT / "out/figures"))
    a = ap.parse_args()
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    static = json.load(open(ROOT / "data/static.json"))
    sub, gt = pd.read_csv(a.sub), pd.read_csv(a.gt)
    E = errors(sub, gt)
    fig_trajectory(sub, gt, E, static, out)
    fig_error_time(E, pd.read_csv(a.log) if a.log else None, out)
    fig_error_map(E, static, out)
    if a.ablation:
        fig_ablation(pd.read_csv(a.ablation), out)
    fig_pipeline(out)
    print(f"figures -> {out}")


if __name__ == "__main__":
    main()
