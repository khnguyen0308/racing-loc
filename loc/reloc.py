"""INTRO method 3a, part 2: absolute pose of the car from one image set and the training-run map.

Inference-safe: uses only the images, calibration, the map built by build_map.py (from training laps)
and the PREDICTED pose of the filter (to pick nearby keyframes). No GT of the run being localized.

    rl = Relocalizer(static, "data/map/laps1-5")                      # cameras fr + cr (vision.CAMS)
    res = rl.locate({"fr": img_fr, "cr": img_cr}, pred=(x, y, yaw))   # list of per-camera fixes
"""
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import cv2
from scipy.spatial import cKDTree

from loc.vision import CAMS, Camera, MapGround, features, match, yaw_of


class Relocalizer:
    def __init__(self, static, map_prefix, cams=CAMS, radius=15.0, max_dheading_deg=30.0,
                 n_kf=3, min_inliers=25, reproj_px=3.0, max_dz=1.0):
        self.cams, self.maps = {}, {}
        for c in cams:
            self.cams[c] = Camera(static, c)
            p = Path(f"{map_prefix}_{c}.npz")
            if not p.exists():
                raise FileNotFoundError(f"no map for camera {c}: {p} "
                                        f"(build it with loc/build_map.py --cams {' '.join(cams)})")
            m = dict(np.load(p))
            m["tree"] = cKDTree(m["pose"][:, :2])
            self.maps[c] = m
        self.ground = MapGround(static)
        self.radius, self.max_dh = radius, np.radians(max_dheading_deg)
        self.n_kf, self.min_inliers, self.reproj_px, self.max_dz = n_kf, min_inliers, reproj_px, max_dz
        self._pool = ThreadPoolExecutor(max_workers=len(cams))

    def _candidates(self, c, x, y, yaw):
        m = self.maps[c]
        ids = np.array(m["tree"].query_ball_point([x, y], self.radius), int)
        if len(ids) == 0:
            return ids
        dh = np.abs(np.angle(np.exp(1j * (m["pose"][ids, 3] - yaw))))
        ids = ids[dh < self.max_dh]                                          # same driving direction
        d = np.hypot(m["pose"][ids, 0] - x, m["pose"][ids, 1] - y)
        return ids[np.argsort(d)][:self.n_kf]

    def locate_cam(self, c, img, pred, feats=None):
        cam, m = self.cams[c], self.maps[c]
        x, y, yaw = pred
        kf = self._candidates(c, x, y, yaw)
        if len(kf) == 0:
            return None
        sl = np.concatenate([np.arange(m["ptr"][k], m["ptr"][k + 1]) for k in kf])
        if len(sl) < self.min_inliers:
            return None
        uv, d = feats if feats is not None else features(img, cam)
        iq, im = match(d, m["D"][sl].astype(np.float32))
        if len(iq) < self.min_inliers:
            return None
        X = m["X"][sl][im].astype(np.float64)
        n = cam.undistort(uv[iq])
        cv2.setRNGSeed(0)                # deterministic RANSAC (OpenCV RNG is per thread): reproducible CSV
        ok, rvec, tvec, inl = cv2.solvePnPRansac(X, n, np.eye(3), None, iterationsCount=300,
                                                 reprojectionError=self.reproj_px / cam.K[0, 0],
                                                 confidence=0.999, flags=cv2.SOLVEPNP_EPNP)
        if not ok or inl is None or len(inl) < self.min_inliers:
            return None
        inl = inl.ravel()
        rvec, tvec = cv2.solvePnPRefineLM(X[inl], n[inl], np.eye(3), None, rvec, tvec)
        R_cm = cv2.Rodrigues(rvec)[0]
        T_cm = np.eye(4); T_cm[:3, :3] = R_cm; T_cm[:3, 3] = tvec.ravel()
        T_mb = np.linalg.inv(T_cm) @ np.linalg.inv(cam.T_bc)                  # base_link in map
        bx, by, bz = T_mb[:3, 3]; byaw = yaw_of(T_mb)
        zg, _, _ = self.ground.attitude(bx, by, byaw)
        if abs(bz - zg) > self.max_dz:                                       # car must sit on the road
            return None
        e = np.linalg.norm((X[inl] @ R_cm.T + tvec.ravel())[:, :2] / (X[inl] @ R_cm.T + tvec.ravel())[:, 2:] - n[inl], axis=1)
        return dict(cam=c, x=bx, y=by, z=bz, yaw=byaw, n_inl=len(inl), n_match=len(iq),
                    reproj_px=float(np.median(e) * cam.K[0, 0]), kf=kf)

    def locate(self, imgs, pred):
        """imgs: {cam: BGR image}. Returns list of per-camera fixes (may be empty).
        Cameras run in parallel threads (OpenCV SIFT / matching release the GIL)."""
        items = [(c, img) for c, img in imgs.items() if c in self.cams and img is not None]
        if len(items) > 1:
            res = list(self._pool.map(lambda ci: self.locate_cam(ci[0], ci[1], pred), items))
        else:
            res = [self.locate_cam(c, img, pred) for c, img in items]
        return [r for r in res if r is not None]
