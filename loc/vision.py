"""Shared camera geometry + features for map building (build_map.py) and relocalization (reloc.py).

Frames: map; base_link (x fwd, y left, z up); camera (OpenCV: x right, y down, z forward).
T_a_b = 4x4 pose of frame b expressed in frame a  ->  p_a = T_a_b @ p_b.
"""
import sys
import threading
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools_team"))
from masks import get_mask  # noqa: E402

CAMS = ("fr", "cr")   # cameras of the map AND of the localization (chosen on lap 5): one default for every script


def rot_zyx(yaw, pitch, roll):
    cy, sy, cp, sp, cr, sr = np.cos(yaw), np.sin(yaw), np.cos(pitch), np.sin(pitch), np.cos(roll), np.sin(roll)
    return np.array([[cy*cp, cy*sp*sr - sy*cr, cy*sp*cr + sy*sr],
                     [sy*cp, sy*sp*sr + cy*cr, sy*sp*cr - cy*sr],
                     [-sp,   cp*sr,            cp*cr]])


def T_from(R, t):
    T = np.eye(4); T[:3, :3] = R; T[:3, 3] = t
    return T


def T_map_base(x, y, z, yaw, pitch=0.0, roll=0.0):
    return T_from(rot_zyx(yaw, pitch, roll), [x, y, z])


def yaw_of(T):
    return float(np.arctan2(T[1, 0], T[0, 0]))


class Camera:
    def __init__(self, static, cam, mask_erode=12):
        c = static["cameras"][cam]
        self.name, self.W, self.H = cam, c["width"], c["height"]
        self.K, self.D = np.array(c["K"], float), np.array(c["D"], float)
        self.T_bc = np.array(c["T_base_cam"], float)              # camera pose in base_link
        m = get_mask(cam, (self.H, self.W), mask_dir=str(ROOT / "data/masks"))
        # erode: a SIFT descriptor near the mask border would still see the car body
        self.mask = cv2.erode(m, np.ones((2 * mask_erode + 1,) * 2, np.uint8))

    def T_map_cam(self, T_mb):
        return T_mb @ self.T_bc

    def undistort(self, uv):
        """pixels (N,2) -> normalized image coordinates (N,2) (z = 1 plane, no distortion)."""
        if len(uv) == 0:
            return np.zeros((0, 2))
        return cv2.undistortPoints(np.asarray(uv, np.float64).reshape(-1, 1, 2), self.K, self.D).reshape(-1, 2)

    def project(self, X, T_mc):
        """map points (N,3) -> pixels (N,2), depth (N,)."""
        if len(X) == 0:
            return np.zeros((0, 2)), np.zeros(0)
        T_cm = np.linalg.inv(T_mc)
        Xc = X @ T_cm[:3, :3].T + T_cm[:3, 3]
        uv, _ = cv2.projectPoints(Xc.reshape(-1, 1, 3), np.zeros(3), np.zeros(3), self.K, self.D)
        return uv.reshape(-1, 2), Xc[:, 2]


class MapGround:
    """Road height, pitch and roll under the car, from the 3D map borders only (inference-safe).
    Checked on laps 1-5: z within ~5 cm, pitch p95 0.5 deg, roll p95 0.9 deg of GT."""
    def __init__(self, static):
        from scipy.spatial import cKDTree
        tb = static["track_bounds"]
        self.L = np.array(tb["track_left"]["points"], float)
        self.R = np.array(tb["track_right"]["points"], float)
        self.tL, self.tR = cKDTree(self.L[:, :2]), cKDTree(self.R[:, :2])

    def attitude(self, x, y, yaw, look=4.0):
        """z of the road centre, pitch, roll (rad, rot_zyx convention) at (x, y, yaw)."""
        c, s = np.cos(yaw), np.sin(yaw)
        pts = np.array([[x, y], [x + look * c, y + look * s], [x - look * c, y - look * s]])
        l, r = self.L[self.tL.query(pts)[1]], self.R[self.tR.query(pts)[1]]
        zc = (l[:, 2] + r[:, 2]) / 2
        pitch = -np.arctan2(zc[1] - zc[2], 2 * look)                     # uphill -> nose up -> pitch < 0
        lr = l[0] - r[0]
        side = np.sign(-s * lr[0] + c * lr[1])
        roll = np.arctan2(lr[2], np.hypot(lr[0], lr[1])) * side           # left side higher -> roll > 0
        return float(zc[0]), float(pitch), float(roll)


_tls = threading.local()      # OpenCV CLAHE / SIFT / BFMatcher objects are not thread-safe: one set per thread


def _cv():
    if not hasattr(_tls, "sift"):
        _tls.clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        _tls.sift = cv2.SIFT_create(nfeatures=2000, contrastThreshold=0.02)   # low threshold: dark night images
        _tls.bf = cv2.BFMatcher(cv2.NORM_L2)
    return _tls


def features(img, cam):
    """SIFT on the CLAHE-enhanced gray image, car body masked. -> pixels (N,2) float32, desc (N,128) float32."""
    cv = _cv()
    g = cv.clahe.apply(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if img.ndim == 3 else img)
    kp, des = cv.sift.detectAndCompute(g, cam.mask)
    if des is None:
        return np.zeros((0, 2), np.float32), np.zeros((0, 128), np.float32)
    return np.float32([k.pt for k in kp]), des


def match(d1, d2, ratio=0.8):
    """Lowe ratio test, mutual-unique on d2. -> (i1, i2) index arrays."""
    if len(d1) < 2 or len(d2) < 2:
        return np.zeros(0, int), np.zeros(0, int)
    mm = _cv().bf.knnMatch(np.asarray(d1, np.float32), np.asarray(d2, np.float32), k=2)
    good = [(m.queryIdx, m.trainIdx, m.distance) for m, n in (p for p in mm if len(p) == 2) if m.distance < ratio * n.distance]
    best = {}
    for q, t, d in good:                                           # one query per train point
        if t not in best or d < best[t][1]:
            best[t] = (q, d)
    if not best:
        return np.zeros(0, int), np.zeros(0, int)
    i2 = np.fromiter(best.keys(), int); i1 = np.array([best[t][0] for t in i2])
    return i1, i2
