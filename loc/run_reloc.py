#!/usr/bin/env python3
"""Causal EKF (INTRO method 4) + map relocalization (method 3a) -> pose CSV, one pose per image set.

    # 10 Hz, images decoded straight from the mcap (what run.py uses)
    python loc/run_reloc.py --mcap data/c0_2026-10-13_22-53-42_camloc_0.mcap --gt data/gt_val.csv \
        --map data/map/laps1-5 --calib config/motion_calib_laps1-5.json --out out/reloc_val.csv
    # 5 Hz from the JPEGs of extract_images.py (fast development)
    python loc/run_reloc.py --gt data/gt_lap5.csv --map data/map/laps1-4 --out out/reloc_lap5.csv

Per image set at stamp t (header.stamp of cams[0]): predict with speed + yaw rate (state stamps <= t),
relocalize every camera around the PREDICTED pose, update the EKF (chi2 gate + re-lock), write the pose.
--gt / --first-pose: ONLY the first pose and the time window. The map must not contain the scored laps.
"""
import argparse, json, sys, time
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "tools_team")]
from loc.ekf import CarEKF, body_cov, wrap  # noqa: E402
from loc.reloc import Relocalizer           # noqa: E402
from loc.vision import CAMS                 # noqa: E402
from state_io import load_state             # noqa: E402


def mcap_source(mcap, cams, t0, t1, max_wait_ns=50_000_000):
    """Yields (t, {cam: BGR image}) for every image set with stamp in [t0, t1], in time order.
    A set is the images of all cams within 1 ms (hardware sync); incomplete sets are still yielded."""
    from mcap_utils import img_topic, iter_msgs, stamp_ns
    topics = {img_topic(c): c for c in cams}
    pending = {}                                                      # stamp of cams[0]-group -> {cam: img}

    def flush(older_than):
        for k in sorted(k for k in pending if k < older_than):
            yield k, pending.pop(k)

    for topic, _, msg in iter_msgs(mcap, list(topics), start_ns=t0 - 200_000_000, end_ns=t1 + 500_000_000):
        t = stamp_ns(msg.header)
        if t < t0 - 1_000_000 or t > t1 + 1_000_000:
            continue
        key = next((k for k in pending if abs(k - t) < 1_000_000), t)
        pending.setdefault(key, {})[topics[topic]] = cv2.imdecode(np.frombuffer(bytes(msg.data), np.uint8), cv2.IMREAD_COLOR)
        if len(pending[key]) == len(cams):
            yield from flush(key)                                     # older incomplete sets first
            yield key, pending.pop(key)
        yield from flush(t - max_wait_ns)
    yield from flush(np.iinfo(np.int64).max)


def jpeg_source(images, stamps_csv, cams, t0, t1, ref="fl"):
    """Yields (t, {cam: image} or None) at every `ref` camera stamp; images where JPEGs were saved (5 Hz)."""
    idx = pd.read_csv(Path(images) / "index.csv"); idx = idx[idx.file.notna() & (idx.stamp_ns >= t0 - 1e6) & (idx.stamp_ns <= t1 + 1e6)]
    per_cam = {c: (idx[idx.cam == c].stamp_ns.to_numpy(np.int64), idx[idx.cam == c].file.to_numpy()) for c in cams}
    st = pd.read_csv(stamps_csv)
    for t in np.sort(st[st.cam == ref].stamp_ns.to_numpy(np.int64)):
        if t < t0 or t > t1:
            continue
        imgs = {}
        for c, (s, f) in per_cam.items():
            if len(s):
                k = np.argmin(np.abs(s - t))
                if abs(s[k] - t) < 1_000_000:
                    imgs[c] = cv2.imread(str(Path(images) / f[k]))
        yield int(t), (imgs or None)


def localize(source, state, first_pose, t0, rl, yawrate_bias=0.0, sig_lat=0.25, sig_lon=0.4, sig_yaw_deg=0.2,
             relock=5, verbose_every=0):
    """The causal filter loop. source yields (t, imgs or None) in time order. Returns (poses df, log df, ms list)."""
    ts, v, wz = state.timestamp_ns.to_numpy(np.int64), state.v_mps.to_numpy(float), state.wz_radps.to_numpy(float)
    # body lateral speed (sideslip): halves the drift in tight corners between camera fixes (lap 5: 1.55 -> 0.66 m / 3 s)
    vy = state.vy_mps.to_numpy(float) if "vy_mps" in state else np.zeros(len(state))
    f = CarEKF(*first_pose)                                           # first pose: the only GT used
    f.X[3] = yawrate_bias                                             # calibrated on training laps
    j = max(np.searchsorted(ts, t0, side="right") - 1, 0); t_cur = t0
    out, log, ms, rej = [], [], [], []                                # rej: recent rejected fixes
    for n, (t, imgs) in enumerate(source):
        while j + 1 < len(ts) and ts[j + 1] <= t:                     # causal predict up to t
            f.predict(v[j], wz[j], (ts[j + 1] - t_cur) / 1e9, vy[j]); t_cur = ts[j + 1]; j += 1
        f.predict(v[j], wz[j], (t - t_cur) / 1e9, vy[j]); t_cur = max(t_cur, t)
        if imgs and rl is not None:
            tic = time.perf_counter()
            fixes = rl.locate(imgs, tuple(f.X[:3]))
            for q in fixes:
                R = body_cov(q["yaw"], sig_lon, sig_lat, np.radians(sig_yaw_deg))
                ok, d2 = f.update_pose((q["x"], q["y"], q["yaw"]), R)
                re = False
                if ok:
                    rej = []
                else:
                    # rejected fixes, moved to "now" with the motion since they were taken (filter-relative)
                    rej = [r for r in rej if t - r[0] < 2e9] + [(t, q["x"] - f.X[0], q["y"] - f.X[1])]
                    if relock and len(rej) >= relock:
                        off = np.array([[r[1], r[2]] for r in rej[-relock:]])
                        spread = np.max(np.linalg.norm(off - np.median(off, 0), axis=1))
                        if spread < 1.0:                  # lock-out: N independent fixes agree, the filter is wrong
                            f.P[0, 0] += 3.0 ** 2; f.P[1, 1] += 3.0 ** 2; f.P[2, 2] += np.radians(1.0) ** 2
                            ok, d2 = f.update_pose((q["x"], q["y"], q["yaw"]), R)
                            re, rej = True, []
                log.append(dict(timestamp_ns=t, cam=q["cam"], used=ok, d2=d2, inliers=q["n_inl"], relock=re))
            if not fixes:
                log.append(dict(timestamp_ns=t, cam="", used=False, d2=np.nan, inliers=0, relock=False))
            ms.append(1000 * (time.perf_counter() - tic))
        out.append((t, f.X[0], f.X[1], wrap(f.X[2]), np.sqrt(f.P[0, 0] + f.P[1, 1])))
        if verbose_every and n % verbose_every == 0:
            print(f"  frame {n}: {np.mean(ms[-verbose_every:]) if ms else 0:.0f} ms/frame, sigma_pos {out[-1][4]:.2f} m", flush=True)
    D = pd.DataFrame(out, columns=["timestamp_ns", "x", "y", "yaw", "sigma_pos"])
    return D, pd.DataFrame(log), ms, f


def summary(D, G, ms, f):
    s = f"{len(D)} poses"
    if len(G):
        fx = G.cam != ""
        s += (f" | fixes {int(fx.sum())}, used {int(G.used.sum())}, gated out {int((fx & ~G.used).sum())}, "
              f"re-locks {int(G.relock.eq(True).sum())}")
    if ms:
        s += f" | {np.mean(ms):.0f} ms per image set (p95 {np.percentile(ms, 95):.0f})"
    return s + f" | final bias {f.X[3]:+.6f} rad/s, k_v {f.X[4]:.4f}"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--map", required=True); ap.add_argument("--out", required=True)
    ap.add_argument("--gt", help="CSV whose FIRST row is the first pose and whose span is the time window")
    ap.add_argument("--mcap", help="read images at 10 Hz from this mcap (default: JPEGs at 5 Hz)")
    ap.add_argument("--state", default=str(ROOT / "data/state_full.csv"))
    ap.add_argument("--stamps", default=str(ROOT / "data/images/index.csv"), help="JPEG mode: output stamps")
    ap.add_argument("--images", default=str(ROOT / "data/map_images"))
    ap.add_argument("--static", default=str(ROOT / "data/static.json"))
    ap.add_argument("--calib", default=str(ROOT / "config/motion_calib.json"))
    ap.add_argument("--cams", nargs="+", default=list(CAMS))
    ap.add_argument("--no-camera", action="store_true", help="prediction only (B0 + calibrated bias)")
    ap.add_argument("--no-vy", action="store_true", help="ignore the body lateral speed vy_mps in the prediction")
    ap.add_argument("--relock", type=int, default=5,
                    help="after N fixes in a row rejected by the gate, inflate the filter uncertainty (0 = off)")
    a = ap.parse_args()

    gt = pd.read_csv(a.gt)
    t0, t1 = int(gt.timestamp_ns.iloc[0]), int(gt.timestamp_ns.iloc[-1])
    L = pd.read_csv(ROOT / "data/laps.csv")
    scored = set(L[(L.end_ns > t0) & (L.start_ns < t1)].lap.tolist())
    map_laps = set(np.load(f"{a.map}_{a.cams[0]}.npz")["laps"].tolist())
    if not a.no_camera and scored & map_laps:
        raise SystemExit(f"map {a.map} contains scored laps {sorted(scored & map_laps)}: not a fair test")
    calib = json.load(open(a.calib))
    S = load_state(a.state, ["v_mps", "wz_radps"] + ([] if a.no_vy else ["vy_mps"]))
    rl = None if a.no_camera else Relocalizer(json.load(open(a.static)), a.map, cams=a.cams)
    src = (mcap_source(a.mcap, a.cams, t0, t1) if a.mcap else jpeg_source(a.images, a.stamps, a.cams, t0, t1))
    D, G, ms, f = localize(src, S, (gt.x.iloc[0], gt.y.iloc[0], gt.yaw.iloc[0]), t0, rl, calib["yawrate_bias"],
                           relock=a.relock, verbose_every=500)
    D[["timestamp_ns", "x", "y", "yaw"]].to_csv(a.out, index=False, float_format="%.6f")
    G.to_csv(a.out.replace(".csv", "_log.csv"), index=False)
    print(summary(D, G, ms, f), "->", a.out)


if __name__ == "__main__":
    main()
