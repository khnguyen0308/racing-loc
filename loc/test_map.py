#!/usr/bin/env python3
"""DEV CHECK (reads GT) for method 3a.

    python loc/test_map.py reproj --cam fl --n 40     # 3.3: triangulate (i, i+1), check in the unseen frame i+2
    python loc/test_map.py reloc --map data/map/laps1-4 --lap 5 --n 150   # 3.4: PnP on a lap the map never saw

Pass 3.3: median reprojection error in frame i+2 < 2-3 px (frames / signs / calibration are right).
Pass 3.4: > 70 % of frames with a fix, median error < 0.5 m and < 0.5 deg.
"""
import argparse, json, sys
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT)]
from loc.build_map import gt_poses, triangulate_pair  # noqa: E402
from loc.vision import CAMS, Camera, features, match   # noqa: E402


def test_reproj(cam_name, n, laps=(1, 2, 3, 4)):
    static = json.load(open(ROOT / "data/static.json")); gt = pd.read_csv(ROOT / "data/gt.csv")
    S = pd.read_csv(ROOT / "data/state_full.csv", usecols=["timestamp_ns", "pitch_rad", "roll_rad", "v_mps"])
    L = pd.read_csv(ROOT / "data/laps.csv"); L = L[L.lap.isin(laps)]
    idx = pd.read_csv(ROOT / "data/map_images/index.csv"); idx = idx[(idx.cam == cam_name) & idx.file.notna()].sort_values("stamp_ns")
    idx = idx[(idx.stamp_ns >= L.start_ns.min()) & (idx.stamp_ns < L.end_ns.max())]
    idx = idx[np.interp(idx.stamp_ns, S.timestamp_ns, S.v_mps) > 8].reset_index(drop=True)
    cam = Camera(static, cam_name)
    errs, npts, nmatch = [], [], []
    for i in np.linspace(0, len(idx) - 3, n).astype(int):
        rows = idx.iloc[i:i + 3]
        if (rows.stamp_ns.iloc[2] - rows.stamp_ns.iloc[0]) > 0.5e9:
            continue
        T = [cam.T_map_cam(Tb) for Tb in gt_poses(rows.stamp_ns.to_numpy(np.int64), gt, S)]
        F = []
        for f in rows.file:
            uv, d = features(cv2.imread(str(ROOT / "data/map_images" / f)), cam)
            F.append((uv, d, cam.undistort(uv)))
        ii, X = triangulate_pair(cam, F[0], F[1], T[0], T[1])
        npts.append(len(X))
        if len(X) == 0:
            continue
        a, b = match(F[0][1][ii], F[2][1])                 # same physical point found again in frame i+2?
        nmatch.append(len(a))
        uv, z = cam.project(X[a], T[2])
        e = np.linalg.norm(uv - F[2][0][b], axis=1)[z > 0]
        errs.extend(e)
    e = np.array(errs)
    print(f"{cam_name}: {len(npts)} triples | 3D points per pair median {np.median(npts):.0f} | "
          f"re-found in frame i+2 median {np.median(nmatch):.0f}")
    print(f"reprojection error in unseen frame i+2: median {np.median(e):.2f} px, p75 {np.percentile(e, 75):.2f}, "
          f"<3 px {100 * np.mean(e < 3):.0f} %, >20 px (wrong matches) {100 * np.mean(e > 20):.0f} %")


def test_reloc(map_prefix, lap, n, cams):
    """3.4: map from other laps, frames of `lap`; prediction = GT + offset. Errors vs GT per camera fix."""
    import time
    from loc.reloc import Relocalizer
    static = json.load(open(ROOT / "data/static.json")); gt = pd.read_csv(ROOT / "data/gt.csv")
    L = pd.read_csv(ROOT / "data/laps.csv"); r0 = L[L.lap == lap].iloc[0]
    if lap in np.load(f"{map_prefix}_{cams[0]}.npz")["laps"]:
        raise SystemExit(f"lap {lap} is inside the map: test on a lap the map has not seen")
    idx = pd.read_csv(ROOT / "data/map_images/index.csv"); idx = idx[idx.file.notna()]
    ref = idx[(idx.cam == cams[0]) & (idx.stamp_ns >= r0.start_ns) & (idx.stamp_ns < r0.end_ns)].sort_values("stamp_ns")
    ref = ref.iloc[np.linspace(0, len(ref) - 1, n).astype(int)]
    rl = Relocalizer(static, map_prefix, cams=cams)
    tg = gt.timestamp_ns.to_numpy(np.int64); yaw_u = np.unwrap(gt.yaw.to_numpy())
    offsets = [("exact", 0, 0, 0), ("lat 2 m + yaw 1 deg", 0, 2, 1), ("along 5 m", 5, 0, 0)]
    rows, tt = [], []
    for r in ref.itertuples():
        imgs = {}
        for c in cams:
            cand = idx[idx.cam == c]; k = np.argmin(np.abs(cand.stamp_ns.to_numpy() - r.stamp_ns))
            imgs[c] = cv2.imread(str(ROOT / "data/map_images" / cand.file.iloc[k]))
        gx, gy, gyaw = np.interp(r.stamp_ns, tg, gt.x), np.interp(r.stamp_ns, tg, gt.y), np.interp(r.stamp_ns, tg, yaw_u)
        c_, s_ = np.cos(gyaw), np.sin(gyaw)
        for name, dlon, dlat, dyaw in offsets:
            pred = (gx + c_ * dlon - s_ * dlat, gy + s_ * dlon + c_ * dlat, gyaw + np.radians(dyaw))
            tic = time.perf_counter(); res = rl.locate(imgs, pred); tt.append(time.perf_counter() - tic)
            for c in cams:
                f = [q for q in res if q["cam"] == c]
                if not f:
                    rows.append(dict(t=r.stamp_ns, gx=gx, gy=gy, off=name, cam=c, ok=False)); continue
                f = f[0]; dx, dy = f["x"] - gx, f["y"] - gy
                rows.append(dict(t=r.stamp_ns, gx=gx, gy=gy, off=name, cam=c, ok=True, pos=np.hypot(dx, dy),
                                 lat=-s_ * dx + c_ * dy, lon=c_ * dx + s_ * dy,
                                 yaw=np.degrees(np.angle(np.exp(1j * (f["yaw"] - gyaw)))), inl=f["n_inl"]))
    D = pd.DataFrame(rows)
    print(f"map {map_prefix}, lap {lap}: {len(ref)} frames, {1000 * np.mean(tt):.0f} ms per image set ({'+'.join(cams)})")
    print(f"{'prediction':22s} {'cam':4s} {'fix %':>6s} {'pos med':>8s} {'pos p90':>8s} {'|lat| med':>9s} {'|lon| med':>9s} "
          f"{'|yaw| med':>9s} {'<0.5 m %':>9s} {'>3 m %':>7s} {'inliers':>7s}")
    for (o, c), g in D.groupby(["off", "cam"], sort=False):
        k = g[g.ok]
        if len(k) == 0:
            print(f"{o:22s} {c:4s} {0:6.0f}"); continue
        print(f"{o:22s} {c:4s} {100 * len(k) / len(g):6.0f} {k.pos.median():8.2f} {k.pos.quantile(0.9):8.2f} "
              f"{k.lat.abs().median():9.2f} {k.lon.abs().median():9.2f} {k.yaw.abs().median():9.2f} "
              f"{100 * (k.pos < 0.5).mean():9.0f} {100 * (k.pos > 3).mean():7.0f} {k.inl.median():7.0f}"
              f"   signed med: lat {k.lat.median():+.2f} lon {k.lon.median():+.2f} yaw {k.yaw.median():+.2f}")
    D.to_csv(ROOT / f"out/reloc_test_lap{lap}_{'+'.join(cams)}.csv", index=False)
    # complementarity: does another camera give a good fix where the first camera fails?
    E = D[D.off == "exact"]
    good = E.assign(g=E.ok & (E.pos.fillna(99) < 1.0)).pivot_table(index="t", columns="cam", values="g", aggfunc="first")
    base = cams[0]
    print(f"good fix (< 1 m), exact prediction: " + ", ".join(f"{c} {100 * good[c].mean():.0f} %" for c in cams)
          + f" | any camera {100 * good.any(axis=1).mean():.0f} %")
    for c in cams[1:]:
        miss = ~good[base]
        print(f"  where {base} fails ({int(miss.sum())} frames): {c} rescues {100 * good.loc[miss, c].mean():.0f} %")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("test", choices=["reproj", "reloc"])
    ap.add_argument("--cam", default="fl"); ap.add_argument("--n", type=int, default=40)
    ap.add_argument("--map", default=str(ROOT / "data/map/laps1-4")); ap.add_argument("--lap", type=int, default=5)
    ap.add_argument("--cams", nargs="+", default=list(CAMS))
    a = ap.parse_args()
    if a.test == "reproj":
        test_reproj(a.cam, a.n)
    else:
        test_reloc(a.map, a.lap, a.n, a.cams)


if __name__ == "__main__":
    main()
