#!/usr/bin/env python3
"""STEP 5 - project the map track borders into the camera images with the GT pose.
(Development tool: uses GT. Purpose: check / refine the camera calibration and later make labels.)

    # overlays for 12 images of camera cr spread over laps 1-5
    python3 tools_team/project_borders.py --cam cr --n 12 --out out/overlay_cr
    # same + automatic refinement of camera yaw/pitch/roll (edge alignment), saved to config/
    python3 tools_team/project_borders.py --cam cr --n 40 --refine --out out/overlay_cr

Inputs (defaults): data/static.json, data/gt.csv, data/state_full.csv (roll/pitch of the car, dev only),
data/images/index.csv + data/images/<cam>/<stamp>.jpg (from extract_images.py), data/laps.csv
"""
import argparse, json, sys
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

sys.path[:0] = [str(Path(__file__).resolve().parent), str(Path(__file__).resolve().parents[2] / "tools_team")]
from masks import get_mask  # noqa: E402

COLORS = {"track_left": (255, 80, 30), "track_right": (40, 40, 255),
          "pitlane_left": (255, 200, 120), "pitlane_right": (140, 140, 255)}


def rot_zyx(yaw, pitch, roll):
    cy, sy, cp, sp, cr, sr = np.cos(yaw), np.sin(yaw), np.cos(pitch), np.sin(pitch), np.cos(roll), np.sin(roll)
    return np.array([[cy*cp, cy*sp*sr - sy*cr, cy*sp*cr + sy*sr],
                     [sy*cp, sy*sp*sr + cy*cr, sy*sp*cr - cy*sr],
                     [-sp,   cp*sr,            cp*cr]])


def correction(dyaw_deg=0.0, dpitch_deg=0.0, droll_deg=0.0):
    """Small rotation applied in the CAMERA optical frame (x right, y down, z forward):
    yaw = about y (turn left/right), pitch = about x (tilt up/down), roll = about z."""
    a, b, c = np.radians([dpitch_deg, dyaw_deg, droll_deg])
    Rx = np.array([[1, 0, 0], [0, np.cos(a), -np.sin(a)], [0, np.sin(a), np.cos(a)]])
    Ry = np.array([[np.cos(b), 0, np.sin(b)], [0, 1, 0], [-np.sin(b), 0, np.cos(b)]])
    Rz = np.array([[np.cos(c), -np.sin(c), 0], [np.sin(c), np.cos(c), 0], [0, 0, 1]])
    return Ry @ Rx @ Rz


class Projector:
    def __init__(self, static, cam, max_range=80.0):
        c = static["cameras"][cam]
        self.K, self.D = np.array(c["K"]), np.array(c["D"])
        self.T_bc = np.array(c["T_base_cam"])
        self.W, self.H = c["width"], c["height"]
        self.borders = {k: np.array(v["points"], float) for k, v in static["track_bounds"].items()}
        self.max_range = max_range

    def project(self, pose, corr=np.eye(3), dz=0.0):
        """pose = (x, y, z, yaw, pitch, roll) of base_link in map. Returns {name: list of Nx2 pixel arrays}."""
        x, y, z, yaw, pitch, roll = pose
        R_mb = rot_zyx(yaw, pitch, roll); t_mb = np.array([x, y, z])
        R_bc = self.T_bc[:3, :3] @ corr; t_bc = self.T_bc[:3, 3] + [0, 0, dz]
        R_mc = R_mb @ R_bc; t_mc = t_mb + R_mb @ t_bc
        out = {}
        for name, P in self.borders.items():
            near = np.linalg.norm(P[:, :2] - [x, y], axis=1) < self.max_range
            if not near.any():
                continue
            Pc = (P[near] - t_mc) @ R_mc                       # map -> camera frame
            ok = Pc[:, 2] > 1.0                                # in front of the camera
            # keep only rays inside ~the field of view before applying the distortion polynomial
            ok &= np.abs(Pc[:, 0] / np.maximum(Pc[:, 2], 1e-6)) < 1.2 * self.W / (2 * self.K[0, 0]) + 0.3
            ok &= np.abs(Pc[:, 1] / np.maximum(Pc[:, 2], 1e-6)) < 1.2 * self.H / (2 * self.K[1, 1]) + 0.3
            segs, cur = [], []
            for p, good in zip(Pc, ok):                        # split polylines where points leave the view
                if good:
                    cur.append(p)
                elif cur:
                    segs.append(np.array(cur)); cur = []
            if cur:
                segs.append(np.array(cur))
            px = []
            for s in segs:
                if len(s) < 2:
                    continue
                uv, _ = cv2.projectPoints(s.reshape(-1, 1, 3), np.zeros(3), np.zeros(3), self.K, self.D)
                px.append(uv.reshape(-1, 2))
            out[name] = px
        return out


def gt_pose_at(t_ns, gt, state):
    t = gt.timestamp_ns.to_numpy(np.int64)
    x = np.interp(t_ns, t, gt.x); y = np.interp(t_ns, t, gt.y); z = np.interp(t_ns, t, gt.z)
    yaw = np.interp(t_ns, t, np.unwrap(gt.yaw))
    ts = state.timestamp_ns.to_numpy(np.int64)
    return (x, y, z, yaw, np.interp(t_ns, ts, state.pitch_rad), np.interp(t_ns, ts, state.roll_rad))


def edge_cost(dist_map, proj, H, W, trunc=25.0):
    """Mean (truncated) distance in pixels from projected border points to the nearest image edge."""
    vals = []
    for segs in proj.values():
        for uv in segs:
            uv = uv[(uv[:, 0] >= 0) & (uv[:, 0] < W - 1) & (uv[:, 1] >= 0) & (uv[:, 1] < H - 1)]
            if len(uv):
                vals.append(np.minimum(dist_map[uv[:, 1].astype(int), uv[:, 0].astype(int)], trunc))
    return float(np.mean(np.concatenate(vals))) if vals else trunc


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cam", required=True)
    ap.add_argument("--static", default="data/static.json"); ap.add_argument("--gt", default="data/gt.csv")
    ap.add_argument("--state", default="data/state_full.csv"); ap.add_argument("--images", default="data/images")
    ap.add_argument("--laps", default="data/laps.csv"); ap.add_argument("--train-laps", default="1,2,3,4,5")
    ap.add_argument("--n", type=int, default=12); ap.add_argument("--out", default="out/overlay")
    ap.add_argument("--dyaw", type=float, default=0); ap.add_argument("--dpitch", type=float, default=0)
    ap.add_argument("--droll", type=float, default=0); ap.add_argument("--dz", type=float, default=0)
    ap.add_argument("--refine", action="store_true", help="grid-search yaw/pitch/roll corrections on edges")
    ap.add_argument("--min-speed", type=float, default=8.0)
    a = ap.parse_args()
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    static = json.load(open(a.static)); gt = pd.read_csv(a.gt)
    state = pd.read_csv(a.state, usecols=["timestamp_ns", "roll_rad", "pitch_rad", "v_mps"])
    idx = pd.read_csv(Path(a.images) / "index.csv"); idx = idx[(idx.cam == a.cam) & idx.file.notna() & (idx.file != "")]
    L = pd.read_csv(a.laps); lt = L[L.lap.isin([int(x) for x in a.train_laps.split(",")])]
    inlap = np.zeros(len(idx), bool)
    for r in lt.itertuples():
        inlap |= (idx.stamp_ns.values >= r.start_ns) & (idx.stamp_ns.values < r.end_ns)
    v = np.interp(idx.stamp_ns, state.timestamp_ns, state.v_mps)
    idx = idx[inlap & (v > a.min_speed)]
    pick = idx.iloc[np.linspace(0, len(idx) - 1, min(a.n, len(idx))).astype(int)]
    print(f"{a.cam}: {len(idx)} candidate images in laps {a.train_laps}, using {len(pick)}")
    pr = Projector(static, a.cam)
    frames = []
    for r in pick.itertuples():
        img = cv2.imread(str(Path(a.images) / r.file))
        if img is None:
            continue
        mask = get_mask(a.cam, img.shape)
        edges = cv2.Canny(cv2.GaussianBlur(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY), (5, 5), 0), 30, 90)
        edges[mask == 0] = 0
        dist = cv2.distanceTransform(255 - edges, cv2.DIST_L2, 3)
        frames.append((r.stamp_ns, img, dist, gt_pose_at(r.stamp_ns, gt, state)))

    def total_cost(dy, dp, dr, dz=a.dz):
        C = correction(dy, dp, dr)
        return np.mean([edge_cost(d, pr.project(p, C, dz), pr.H, pr.W) for _, _, d, p in frames])

    best = (a.dyaw, a.dpitch, a.droll)
    c0 = total_cost(*best)
    print(f"edge distance with given calibration: {c0:.2f} px (lower = borders sit on image edges)")
    if a.refine and frames:
        for step, span in [(0.5, 3.0), (0.1, 0.5)]:            # coarse-to-fine grid
            grid = np.arange(-span, span + 1e-9, step)
            for _ in range(2):                                 # coordinate descent yaw -> pitch -> roll
                for k in range(3):
                    cands = []
                    for g in grid:
                        b = list(best); b[k] = best[k] + g
                        cands.append((total_cost(*b), tuple(b)))
                    best = min(cands)[1]
        c1 = total_cost(*best)
        print(f"refined: dyaw {best[0]:+.2f} deg, dpitch {best[1]:+.2f} deg, droll {best[2]:+.2f} deg "
              f"-> edge distance {c0:.2f} -> {c1:.2f} px")
        cfg = Path("config") / "extrinsic_correction.json"
        allc = json.load(open(cfg)) if cfg.exists() else {}
        allc[a.cam] = {"dyaw_deg": best[0], "dpitch_deg": best[1], "droll_deg": best[2], "dz_m": a.dz,
                       "edge_px_before": c0, "edge_px_after": c1, "n_images": len(frames)}
        cfg.parent.mkdir(exist_ok=True); json.dump(allc, open(cfg, "w"), indent=1)
        print(f"-> {cfg}")
    C = correction(*best)
    for k, (t, img, _, pose) in enumerate(frames[:a.n]):
        vis = img.copy()
        for name, segs in pr.project(pose, np.eye(3), a.dz).items():    # thin = original calibration
            for uv in segs:
                cv2.polylines(vis, [uv.astype(np.int32).reshape(-1, 1, 2)], False, (0, 255, 255), 1)
        for name, segs in pr.project(pose, C, a.dz).items():             # thick = corrected
            for uv in segs:
                cv2.polylines(vis, [uv.astype(np.int32).reshape(-1, 1, 2)], False, COLORS.get(name, (0, 255, 0)), 2)
        cv2.putText(vis, f"{a.cam} t={t} yellow=calib  blue/red=corrected", (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        cv2.imwrite(str(out / f"{a.cam}_{k:02d}.jpg"), vis)
    print(f"overlays -> {out}/")


if __name__ == "__main__":
    main()
