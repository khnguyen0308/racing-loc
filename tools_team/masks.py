#!/usr/bin/env python3
"""Car-body masks for the 7 cameras (255 = valid scene pixel, 0 = car body / ignore).

Two sources, combined:
  1. MANUAL polygons (from Nguyen's make_masks.py, drawn on 1506x728 raw frames)
  2. AUTO mask from the data: pixels that barely change over many frames while the car is
     moving are the car itself (body, wing, tyre, lens vignetting). Used to check / extend 1.

Masks live in RAW (distorted) image coordinates. If a module works on undistorted images,
use undistort_mask() with the same K, D, newK as for the image.

    # write data/masks/{cam}_valid.png (+ overlays if JPEGs from extract_images.py exist)
    python3 tools_team/masks.py --images data/images --out data/masks
    # also build the AUTO mask from moving frames and report agreement with the manual one
    python3 tools_team/masks.py --images data/images --out data/masks --auto --state data/state_full.csv

In code:
    from masks import get_mask
    valid = get_mask("cr", shape=img.shape)          # uint8 HxW, 255 = use pixel
"""
import argparse
from pathlib import Path

import cv2
import numpy as np

REFERENCE_SIZE = (1506, 728)  # (width, height) of the raw camera images
CAMERAS = ["fl", "fr", "cl", "cr", "r", "rl", "rr"]

# (x, y) polygons in REFERENCE_SIZE pixels that are EXCLUDED (car body)
EXCLUDED_POLYGONS = {
    "fl": [],
    "fr": [
        [(0, 402), (385, 398), (530, 410), (641, 428), (752, 460), (874, 497), (1003, 528),
         (1120, 541), (1160, 728), (0, 728)],                                     # nose
        [(1137, 555), (1157, 479), (1201, 428), (1285, 402), (1430, 386), (1506, 390),
         (1506, 637), (1425, 626), (1290, 591), (1175, 563)],                      # front-right tyre
    ],
    "cl": [
        [(0, 524), (225, 524), (349, 531), (375, 489), (473, 488), (666, 486), (895, 488),
         (1100, 496), (1310, 502), (1386, 518), (1412, 554), (1506, 492), (1506, 728), (0, 728)],
        [(1091, 499), (1112, 467), (1145, 438), (1180, 419), (1258, 417), (1312, 428),
         (1340, 457), (1353, 510)],                                                # tyre
    ],
    "cr": [
        [(0, 443), (65, 468), (190, 468), (218, 447), (330, 439), (420, 433), (568, 438),
         (817, 447), (1104, 463), (1320, 488), (1506, 500), (1506, 728), (0, 728)],
        [(204, 455), (205, 410), (215, 377), (259, 352), (316, 350), (381, 364), (427, 407),
         (444, 460)],                                                              # tyre
    ],
    "rl": [],
    "rr": [
        [(0, 506), (340, 523), (540, 531), (606, 572), (711, 560), (880, 546), (1090, 534),
         (1240, 527), (1506, 492), (1506, 728), (0, 728)],
        [(1147, 526), (1162, 476), (1208, 447), (1247, 437), (1314, 442), (1358, 470), (1388, 525)],
        [(1391, 343), (1506, 323), (1506, 510), (1448, 507), (1431, 431)],         # wing / suspension
    ],
    "r": [],
}

_CACHE = {}


def manual_mask(cam, shape=(728, 1506), margin=8):
    """Valid-pixel mask from the manual polygons, scaled to `shape` (H, W[, C])."""
    h, w = shape[:2]
    rw, rh = REFERENCE_SIZE
    excl = np.zeros((h, w), np.uint8)
    for poly in EXCLUDED_POLYGONS[cam]:
        pts = np.array([[(round(x * w / rw), round(y * h / rh)) for x, y in poly]], np.int32)
        cv2.fillPoly(excl, pts, 255)
    m = max(1, round(margin * w / rw)) if margin > 0 else 0
    if m and excl.any():
        excl = cv2.dilate(excl, np.ones((2 * m + 1, 2 * m + 1), np.uint8))
    return cv2.bitwise_not(excl)


def get_mask(cam, shape=(728, 1506), mask_dir="data/masks", margin=8):
    """Preferred entry point: uses data/masks/{cam}_valid.png if it exists (manual+auto),
    otherwise the manual polygons. Cached; resized to the requested shape."""
    key = (cam, shape[:2], mask_dir, margin)
    if key not in _CACHE:
        p = Path(mask_dir) / f"{cam}_valid.png"
        if p.exists():
            m = cv2.imread(str(p), cv2.IMREAD_GRAYSCALE)
            if m.shape != tuple(shape[:2]):
                m = cv2.resize(m, (shape[1], shape[0]), interpolation=cv2.INTER_NEAREST)
        else:
            m = manual_mask(cam, shape, margin)
        _CACHE[key] = m
    return _CACHE[key]


def undistort_mask(mask, K, D, newK=None):
    """Same remap as cv2.undistort for the image, nearest-neighbour so it stays binary.
    Pixels that come from outside the raw image are marked invalid."""
    h, w = mask.shape
    K, D = np.asarray(K, float), np.asarray(D, float)
    newK = K if newK is None else np.asarray(newK, float)
    m1, m2 = cv2.initUndistortRectifyMap(K, D, None, newK, (w, h), cv2.CV_32FC1)
    return cv2.remap(mask, m1, m2, cv2.INTER_NEAREST, borderMode=cv2.BORDER_CONSTANT, borderValue=0)


def auto_static_mask(frames, std_thr=6.0, dark_thr=8.0, min_area=2000):
    """frames: list of HxW gray images taken while the car MOVES (scene changes, car does not).
    Pixels with low temporal std (car body) or always ~black (vignetting) -> invalid."""
    st = np.stack([f.astype(np.float32) for f in frames])
    static = (st.std(0) < std_thr) | (st.max(0) < dark_thr)
    static = cv2.morphologyEx(static.astype(np.uint8) * 255, cv2.MORPH_OPEN, np.ones((9, 9), np.uint8))
    n, lab, stats, _ = cv2.connectedComponentsWithStats(static)
    keep = np.zeros_like(static)
    for i in range(1, n):                      # drop small static blobs (e.g. dark sky patches)
        if stats[i, cv2.CC_STAT_AREA] >= min_area:
            keep[lab == i] = 255
    return cv2.bitwise_not(keep)


def overlay(img, valid):
    out = img.copy()
    ex = valid == 0
    out[ex] = (img[ex] * 0.45 + np.array([30, 30, 230]) * 0.55).astype(np.uint8)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--images", default="data/images", help="output dir of extract_images.py")
    ap.add_argument("--out", default="data/masks")
    ap.add_argument("--margin", type=int, default=8)
    ap.add_argument("--auto", action="store_true", help="also build the data-driven mask")
    ap.add_argument("--state", default=None, help="state_full.csv: use frames with v_mps > 10 for --auto")
    ap.add_argument("--n-auto", type=int, default=150)
    a = ap.parse_args()
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    idx = None
    if (Path(a.images) / "index.csv").exists():
        import pandas as pd
        idx = pd.read_csv(Path(a.images) / "index.csv")
        idx = idx[idx.file.notna() & (idx.file != "")]
        if a.state:
            S = pd.read_csv(a.state, usecols=["timestamp_ns", "v_mps"]).sort_values("timestamp_ns")
            j = np.clip(np.searchsorted(S.timestamp_ns.values, idx.stamp_ns.values), 0, len(S) - 1)
            idx = idx[S.v_mps.values[j] > 10]
    for cam in CAMERAS:
        valid = manual_mask(cam, (REFERENCE_SIZE[1], REFERENCE_SIZE[0]), a.margin)
        files = [] if idx is None else list(idx[idx.cam == cam].file)
        if a.auto and len(files) >= 20:
            pick = [files[i] for i in np.linspace(0, len(files) - 1, min(a.n_auto, len(files))).astype(int)]
            frames = [cv2.imread(str(Path(a.images) / f), cv2.IMREAD_GRAYSCALE) for f in pick]
            auto = auto_static_mask([f for f in frames if f is not None])
            inter = np.count_nonzero((valid == 0) & (auto == 0)); union = np.count_nonzero((valid == 0) | (auto == 0))
            print(f"{cam}: manual excl {100*np.mean(valid==0):4.1f}% | auto excl {100*np.mean(auto==0):4.1f}% | IoU {inter/max(union,1):.2f}")
            cv2.imwrite(str(out / f"{cam}_auto_valid.png"), auto)
            valid = cv2.bitwise_and(valid, auto)            # exclude what either says is car
        cv2.imwrite(str(out / f"{cam}_valid.png"), valid)
        print(f"{cam}: {100*np.mean(valid==0):4.1f}% excluded -> {out / (cam + '_valid.png')}")
        if files:
            img = cv2.imread(str(Path(a.images) / files[len(files) // 2]))
            if img is not None:
                cv2.imwrite(str(out / f"{cam}_overlay.jpg"), overlay(img, cv2.resize(valid, (img.shape[1], img.shape[0]), interpolation=cv2.INTER_NEAREST)))


if __name__ == "__main__":
    main()
