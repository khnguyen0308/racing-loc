#!/usr/bin/env python3
"""INTRO method 3a, part 1: build a 3D feature map from the TRAINING run (GT poses allowed here).

    python loc/build_map.py --laps 1 2 3 4 --out data/map/laps1-4      # dev   (test on lap 5)
    python loc/build_map.py --laps 1 2 3 4 5 --out data/map/laps1-5    # final (validate on laps 6-7)

For each camera and each pair of consecutive keyframes (i, i+step):
  SIFT matches -> epipolar check with the KNOWN relative pose (GT) -> triangulation ->
  keep points in front of both cameras, reprojection < 2 px, parallax > 1 deg, depth 3..250 m.
Output <out>_<cam>.npz: keyframe poses + per keyframe the 3D map points and their SIFT descriptors.
"""
import argparse, json, sys, time
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT)]
from loc.vision import CAMS, Camera, T_map_base, features, match  # noqa: E402


def gt_poses(stamps, gt, S):
    """T_map_base at image stamps from GT x, y, z, yaw (+ car pitch/roll, training data only)."""
    tg = gt.timestamp_ns.to_numpy(np.int64); ts = S.timestamp_ns.to_numpy(np.int64)
    yaw = np.unwrap(gt.yaw.to_numpy())
    out = []
    for t in stamps:
        out.append(T_map_base(np.interp(t, tg, gt.x), np.interp(t, tg, gt.y), np.interp(t, tg, gt.z),
                              np.interp(t, tg, yaw), np.interp(t, ts, S.pitch_rad), np.interp(t, ts, S.roll_rad)))
    return out


def triangulate_pair(cam, f1, f2, T1, T2, epi_px=4.0, reproj_px=4.0, min_parallax_deg=0.5, depth=(3.0, 250.0)):
    """f = (uv, desc, norm). T = T_map_cam. Returns (index into f1, X map (N,3))."""
    i1, i2 = match(f1[1], f2[1])
    if len(i1) < 8:
        return np.zeros(0, int), np.zeros((0, 3))
    n1, n2 = f1[2][i1], f2[2][i2]
    C1, C2 = np.linalg.inv(T1), np.linalg.inv(T2)                 # map -> camera
    T21 = C2 @ T1                                                  # camera 1 -> camera 2
    R, t = T21[:3, :3], T21[:3, 3]
    E = np.array([[0, -t[2], t[1]], [t[2], 0, -t[0]], [-t[1], t[0], 0]]) @ R
    h1, h2 = np.c_[n1, np.ones(len(n1))], np.c_[n2, np.ones(len(n2))]
    Ex1, Etx2 = h1 @ E.T, h2 @ E
    num = np.sum(h2 * Ex1, 1)
    sampson = num ** 2 / (Ex1[:, 0] ** 2 + Ex1[:, 1] ** 2 + Etx2[:, 0] ** 2 + Etx2[:, 1] ** 2)
    ok = np.sqrt(sampson) * cam.K[0, 0] < epi_px                   # normalized -> pixels
    if ok.sum() < 8:
        return np.zeros(0, int), np.zeros((0, 3))
    i1, n1, n2 = i1[ok], n1[ok], n2[ok]
    Xh = cv2.triangulatePoints(C1[:3], C2[:3], n1.T, n2.T)
    X = (Xh[:3] / Xh[3]).T
    keep = np.ones(len(X), bool)
    for C, n in ((C1, n1), (C2, n2)):
        Xc = X @ C[:3, :3].T + C[:3, 3]
        z = Xc[:, 2]
        keep &= (z > depth[0]) & (z < depth[1])
        e = np.linalg.norm(Xc[:, :2] / np.maximum(z, 1e-6)[:, None] - n, axis=1) * cam.K[0, 0]
        keep &= e < reproj_px
    r1, r2 = X - T1[:3, 3], X - T2[:3, 3]
    cosang = np.sum(r1 * r2, 1) / (np.linalg.norm(r1, axis=1) * np.linalg.norm(r2, axis=1))
    keep &= np.degrees(np.arccos(np.clip(cosang, -1, 1))) > min_parallax_deg
    return i1[keep], X[keep]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--laps", type=int, nargs="+", required=True)
    ap.add_argument("--cams", nargs="+", default=list(CAMS))
    ap.add_argument("--images", default=str(ROOT / "data/map_images"))
    ap.add_argument("--steps", type=int, nargs="+", default=[3, 1],
                    help="pair keyframe i with i+step (1 = 0.2 s at 5 Hz): long = better depth (tried first), short = more points")
    ap.add_argument("--min-speed", type=float, default=3.0)
    ap.add_argument("--out", required=True)
    ap.add_argument("--allow-val-laps", action="store_true", help="final map for the hidden test run only")
    ap.add_argument("--whole-run", action="store_true",
                    help="every frame of the training run (pit lane, out/in laps too); implies --allow-val-laps")
    a = ap.parse_args()
    a.allow_val_laps |= a.whole_run
    if set(a.laps) & {6, 7} and not a.allow_val_laps:
        raise SystemExit("laps 6 and 7 are validation: never build the map from them (--allow-val-laps: test run only)")
    static = json.load(open(ROOT / "data/static.json"))
    gt = pd.read_csv(ROOT / "data/gt.csv")
    S = pd.read_csv(ROOT / "data/state_full.csv", usecols=["timestamp_ns", "pitch_rad", "roll_rad", "v_mps"])  # GT attitude: map building only
    L = pd.read_csv(ROOT / "data/laps.csv"); L = L[L.lap.isin(a.laps)]
    idx = pd.read_csv(Path(a.images) / "index.csv"); idx = idx[idx.file.notna()]
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    for cn in a.cams:
        cam = Camera(static, cn)
        k = idx[idx.cam == cn].sort_values("stamp_ns")
        inlap = np.zeros(len(k), bool) | a.whole_run
        for r in L.itertuples():
            inlap |= (k.stamp_ns.values >= r.start_ns) & (k.stamp_ns.values < r.end_ns)
        k = k[inlap & (np.interp(k.stamp_ns, S.timestamp_ns, S.v_mps) > a.min_speed)]
        stamps = k.stamp_ns.to_numpy(np.int64); poses = gt_poses(stamps, gt, S)
        T_mc = [cam.T_map_cam(T) for T in poses]
        tic = time.time(); feats = {}                                     # sliding window of features
        files = k.file.tolist()
        X_all, D_all, uv_all, ptr = [], [], [], [0]
        for i in range(len(k)):
            for n in range(i, min(i + max(a.steps) + 1, len(k))):
                if n not in feats:
                    uv, d = features(cv2.imread(str(Path(a.images) / files[n])), cam)
                    feats[n] = (uv, d, cam.undistort(uv))
            if i % 500 == 0:
                print(f"{cn}: keyframe {i}/{len(k)}  ({time.time() - tic:.0f} s)", flush=True)
            Xs, Is = [], []
            for step in a.steps:
                j = i + step
                if j < len(k) and (stamps[j] - stamps[i]) < 0.5e9 * step:   # same continuous run
                    ii, X = triangulate_pair(cam, feats[i], feats[j], T_mc[i], T_mc[j])
                    keep = ~np.isin(ii, np.concatenate(Is)) if Is else np.ones(len(ii), bool)  # one 3D point per keypoint
                    Xs.append(X[keep]); Is.append(ii[keep])
            ii = np.concatenate(Is) if Is else np.zeros(0, int)
            X = np.concatenate(Xs) if Xs else np.zeros((0, 3))
            X_all.append(X.astype(np.float32)); D_all.append(feats[i][1][ii].astype(np.uint8)); uv_all.append(feats[i][0][ii])
            ptr.append(ptr[-1] + len(X))
            del feats[i]
        npts = np.diff(ptr)
        pose6 = np.array([[T[0, 3], T[1, 3], T[2, 3], np.arctan2(T[1, 0], T[0, 0])] for T in poses])
        out = f"{a.out}_{cn}.npz"
        np.savez_compressed(out, stamps=stamps, pose=pose6, T_mc=np.array(T_mc), ptr=np.array(ptr),
                            X=np.concatenate(X_all), D=np.concatenate(D_all), uv=np.concatenate(uv_all),
                            laps=np.array(sorted(set(a.laps) | (set(range(1, 8)) if a.whole_run else set()))))
        print(f"{cn}: {len(k)} keyframes, map points per keyframe median {np.median(npts):.0f} "
              f"(p10 {np.percentile(npts, 10):.0f}), total {ptr[-1]}, {time.time() - tic:.0f} s -> {out}", flush=True)


if __name__ == "__main__":
    main()
