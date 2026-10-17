#!/usr/bin/env python3
"""Video of the estimated path against ground truth (development only: reads GT to show the error).

    python loc/video.py                                   # -> out/video/val_path.mp4 (laps 6-7, ~2 min at 2x)
    python loc/video.py --limit 40 --out out/video/test.mp4

One frame per submitted pose (10 Hz): camera image | map following the car (GT vs estimate) | whole track |
error over time. --fps 20 = 2x real time. Re-encoded to H.264 when ffmpeg is installed (smaller, plays everywhere).
"""
import argparse, json, shutil, subprocess, sys
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "tools_team")]
from loc import plots                  # noqa: E402  palette, rcParams, errors(), borders()
from loc.run_reloc import mcap_source  # noqa: E402
import matplotlib.pyplot as plt        # noqa: E402  (Agg backend set by plots)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sub", default=str(ROOT / "out/submission_val.csv"))
    ap.add_argument("--gt", default=str(ROOT / "data/gt_val.csv"))
    ap.add_argument("--log", default=str(ROOT / "out/run_c0_2026-10-13_22-53-42_camloc_0/reloc_log.csv"))
    ap.add_argument("--mcap", default=str(ROOT / "data/c0_2026-10-13_22-53-42_camloc_0.mcap"))
    ap.add_argument("--cam", default="fr")
    ap.add_argument("--fps", type=float, default=20.0)
    ap.add_argument("--zoom", type=float, default=15.0, help="half width of the map window around the car [m]")
    ap.add_argument("--trail", type=float, default=15.0, help="seconds of path shown in the moving window")
    ap.add_argument("--limit", type=int, default=0, help="stop after N frames (0 = all)")
    ap.add_argument("--out", default=str(ROOT / "out/video/val_path.mp4"))
    a = ap.parse_args()

    sub, gt = pd.read_csv(a.sub), pd.read_csv(a.gt)
    static = json.load(open(ROOT / "data/static.json"))
    E = plots.errors(sub, gt)
    ts = sub.timestamp_ns.to_numpy(np.int64)
    sx, sy = sub.x.to_numpy(), sub.y.to_numpy()
    S = pd.read_csv(ROOT / "data/state_full.csv", usecols=["timestamp_ns", "v_mps"])
    v = np.interp(ts, S.timestamp_ns, S.v_mps)
    fixes = {}                                                    # pose stamp -> cameras with an accepted fix
    if Path(a.log).exists():
        G = pd.read_csv(a.log)
        for t, c in G[G.used.astype(bool) & G.cam.notna()][["timestamp_ns", "cam"]].itertuples(index=False):
            fixes.setdefault(int(t), []).append(c)
    rmse = float(np.sqrt(np.mean(E["pos"] ** 2)))

    W, H = 1280, 720
    fig = plt.figure(figsize=(W / 100, H / 100), dpi=100)
    gs = fig.add_gridspec(2, 3, width_ratios=[2.05, 1, 0.62], height_ratios=[1, 0.6],
                          left=0.03, right=0.985, top=0.86, bottom=0.08, wspace=0.12, hspace=0.22)
    ax_cam, ax_zm, ax_full, ax_err = (fig.add_subplot(gs[0, 0]), fig.add_subplot(gs[0, 1]),
                                      fig.add_subplot(gs[0, 2]), fig.add_subplot(gs[1, :]))
    fig.text(0.03, 0.955, f"Camera-only localization  ·  validation laps 6-7 (never used for training)  ·  "
             f"position RMSE {rmse:.3f} m", fontsize=13, weight="bold", color=plots.INK)
    status = fig.text(0.03, 0.912, "", fontsize=11, color=plots.INK2, family="monospace")

    im = ax_cam.imshow(np.zeros((728, 1506, 3), np.uint8)); ax_cam.axis("off")
    ax_cam.set_title(f"camera {a.cam} (one of the two cameras used)", loc="left", fontsize=10, color=plots.INK2)

    plots.borders(ax_zm, static)
    gt_tr, = ax_zm.plot([], [], color=plots.INK, lw=2.4, label="ground truth", zorder=3)
    es_tr, = ax_zm.plot([], [], color=plots.S1, lw=1.8, ls=(0, (4, 2)), label="estimate", zorder=4)
    gt_pt, = ax_zm.plot([], [], "o", color=plots.INK, ms=9, zorder=5)
    es_pt, = ax_zm.plot([], [], "o", color=plots.S1, ms=6, mec=plots.SURF, mew=1.5, zorder=6)
    ax_zm.set_aspect("equal"); ax_zm.grid(False); ax_zm.set_xticks([]); ax_zm.set_yticks([])
    ax_zm.set_title(f"following the car ({2 * a.zoom:.0f} m window)", loc="left", fontsize=10, color=plots.INK2)
    ax_zm.legend(loc="lower left", fontsize=9)

    plots.borders(ax_full, static, lw=0.5)
    ax_full.plot(E["gx"], E["gy"], color=plots.GRID, lw=1)
    full_pt, = ax_full.plot([], [], "o", color=plots.S1, ms=7, mec=plots.SURF, mew=1.5)
    ax_full.set_aspect("equal"); ax_full.axis("off")
    ax_full.set_title("whole track", loc="left", fontsize=10, color=plots.INK2)

    ax_err.plot(E["t"], E["pos"], color=plots.S1, lw=1.2, label="position error")
    ax_err.plot(E["t"], np.abs(E["lat"]), color=plots.S2, lw=0.8, label="|lateral| error")
    ax_err.axhline(1.0, color=plots.MUTED, lw=1, ls="--")
    ax_err.text(E["t"][-1], 1.03, "target < 1 m", ha="right", va="bottom", color=plots.INK2, fontsize=9)
    cursor = ax_err.axvline(0, color=plots.INK, lw=1.2)
    ax_err.set_xlim(0, E["t"][-1]); ax_err.set_ylim(0, max(1.2, E["pos"].max() * 1.1))
    ax_err.set_title("error against ground truth [m]", loc="left", fontsize=10, color=plots.INK2)
    ax_err.set_xlabel("time since first pose [s]")
    ax_err.legend(loc="upper left", ncol=2, fontsize=9)

    out = Path(a.out); out.parent.mkdir(parents=True, exist_ok=True)
    raw = out.with_name(out.stem + "_mp4v.mp4")
    vw = cv2.VideoWriter(str(raw), cv2.VideoWriter_fourcc(*"mp4v"), a.fps, (W, H))
    n = 0
    for t, imgs in mcap_source(a.mcap, [a.cam], int(ts[0]), int(ts[-1])):
        i = int(np.argmin(np.abs(ts - t)))
        if abs(ts[i] - t) > 2_000_000 or a.cam not in imgs:
            continue
        j = np.searchsorted(ts, ts[i] - int(a.trail * 1e9))
        im.set_data(cv2.cvtColor(imgs[a.cam], cv2.COLOR_BGR2RGB))
        gt_tr.set_data(E["gx"][j:i + 1], E["gy"][j:i + 1]); es_tr.set_data(sx[j:i + 1], sy[j:i + 1])
        gt_pt.set_data([E["gx"][i]], [E["gy"][i]]); es_pt.set_data([sx[i]], [sy[i]]); full_pt.set_data([sx[i]], [sy[i]])
        ax_zm.set_xlim(E["gx"][i] - a.zoom, E["gx"][i] + a.zoom); ax_zm.set_ylim(E["gy"][i] - a.zoom, E["gy"][i] + a.zoom)
        cursor.set_xdata([E["t"][i]] * 2)
        f = fixes.get(int(ts[i]), [])
        status.set_text(f"t {E['t'][i]:6.1f} s   speed {v[i]:5.1f} m/s   error {E['pos'][i]:.2f} m "
                        f"(lateral {abs(E['lat'][i]):.2f} m)   camera fix: {', '.join(f) if f else 'none, prediction only'}")
        fig.canvas.draw()
        vw.write(cv2.cvtColor(np.asarray(fig.canvas.buffer_rgba())[:, :, :3], cv2.COLOR_RGB2BGR))
        n += 1
        if n % 200 == 0:
            print(f"  {n}/{len(ts)} frames", flush=True)
        if a.limit and n >= a.limit:
            break
    vw.release(); plt.close(fig)
    ff = shutil.which("ffmpeg")
    if ff and subprocess.run([ff, "-y", "-loglevel", "error", "-i", str(raw), "-c:v", "libx264", "-crf", "23",
                              "-pix_fmt", "yuv420p", "-movflags", "+faststart", str(out)]).returncode == 0:
        try:
            raw.unlink()
        except OSError:                                           # Windows: file open in a player / scanner
            print(f"note: could not delete {raw}, remove it by hand")
    else:
        raw.replace(out)                                          # no ffmpeg: keep the MPEG-4 part 2 file
    print(f"{n} frames at {a.fps:g} fps ({n / a.fps:.0f} s) -> {out} ({out.stat().st_size / 1e6:.1f} MB)")


if __name__ == "__main__":
    main()
