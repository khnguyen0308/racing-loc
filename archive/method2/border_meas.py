"""Camera measurement from the map track borders (INTRO method 2, render-and-compare).

Inference-safe: uses only images, calibration (static.json), map borders and a PREDICTED pose
(x, y, yaw) from the filter. No ground truth anywhere in this file.

    from loc.border_meas import BorderMeasurer, edge_dist
    bm = BorderMeasurer(static, cams=("cl", "cr", "fr"))
    dists = {c: edge_dist(img[c], c) for c in bm.cams}
    m = bm.measure(dists, pred_xyyaw)      # dict or None (no reliable border match)

Idea: project the 3D border points into each camera at candidate poses around the prediction
(shift across the track, small yaw change) and keep the pose whose projection lies on image edges.
"""
import sys
from pathlib import Path

import cv2
import numpy as np
from scipy.spatial import cKDTree

sys.path[:0] = [str(Path(__file__).resolve().parent), str(Path(__file__).resolve().parents[2] / "tools_team")]
from masks import get_mask          # noqa: E402
from project_borders import rot_zyx  # noqa: E402

TRUNC = 25.0   # px, distance cap: a point far from any edge costs the same, however far


def edge_dist(img, cam):
    """Distance [px] from every pixel to the nearest image edge; car-body edges removed."""
    g = cv2.GaussianBlur(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY), (5, 5), 0)
    e = cv2.Canny(g, 30, 90)
    e[get_mask(cam, img.shape) == 0] = 0
    return cv2.distanceTransform(255 - e, cv2.DIST_L2, 3)


class MapGround:
    """Road height, pitch and roll under the car, from the 3D map borders only."""
    def __init__(self, static):
        tb = static["track_bounds"]
        self.L = np.array(tb["track_left"]["points"], float)
        self.R = np.array(tb["track_right"]["points"], float)
        self.tL, self.tR = cKDTree(self.L[:, :2]), cKDTree(self.R[:, :2])

    def _lr(self, xy):
        return self.L[self.tL.query(xy)[1]], self.R[self.tR.query(xy)[1]]

    def attitude(self, x, y, yaw, look=4.0):
        """z of the road centre, pitch, roll (rad, same convention as rot_zyx) at (x, y, yaw)."""
        c, s = np.cos(yaw), np.sin(yaw)
        pts = np.array([[x, y], [x + look * c, y + look * s], [x - look * c, y - look * s]])
        l, r = self._lr(pts)
        zc = (l[:, 2] + r[:, 2]) / 2
        pitch = -np.arctan2(zc[1] - zc[2], 2 * look)                 # uphill -> nose up -> pitch < 0
        lr = l[0] - r[0]
        side = -s * lr[0] + c * lr[1]                                   # left border lateral offset sign
        roll = np.arctan2(lr[2], np.hypot(lr[0], lr[1])) * np.sign(side)  # left side higher -> roll > 0
        return float(zc[0]), float(pitch), float(roll)


class BorderMeasurer:
    def __init__(self, static, cams=("cl", "cr", "fr"), max_range=25.0, min_range=2.0,
                 lat=np.arange(-1.5, 1.501, 0.1), yaw_deg=np.arange(-1.0, 1.001, 0.25),
                 use_slope=True, z_offset=0.0, trunc=TRUNC, prior_px=0.0):
        self.cams = {}
        for c in cams:
            s = static["cameras"][c]
            self.cams[c] = dict(K=np.array(s["K"], float), D=np.array(s["D"], float),
                                T=np.array(s["T_base_cam"], float), W=s["width"], H=s["height"])
        self.P = np.vstack([np.array(v["points"], float) for v in static["track_bounds"].values()])
        self.tree = cKDTree(self.P[:, :2])
        self.ground = MapGround(static)
        self.max_range, self.min_range = max_range, min_range
        self.lat, self.dyaw = lat, np.radians(yaw_deg)
        self.use_slope, self.z_offset = use_slope, z_offset
        self.trunc = trunc            # px cap per point; small = only edges right on the border count
        self.prior_px = prior_px      # px added at |dlat| = 1 m (quadratic): stay near the prediction

    def _cam_pose(self, cam, x, y, z, yaw, pitch, roll):
        R_mb = rot_zyx(yaw, pitch, roll)
        T = self.cams[cam]["T"]
        return R_mb @ T[:3, :3], np.array([x, y, z]) + R_mb @ T[:3, 3]

    def _project(self, cam, P, R_mc, t_mc):
        """Pixels of map points P (N,3); invalid (behind / far outside the view) -> NaN."""
        c = self.cams[cam]
        Pc = (P - t_mc) @ R_mc
        ok = Pc[:, 2] > 1.0
        z = np.maximum(Pc[:, 2], 1e-6)
        ok &= np.abs(Pc[:, 0] / z) < 1.2 * c["W"] / (2 * c["K"][0, 0]) + 0.3   # keep the distortion
        ok &= np.abs(Pc[:, 1] / z) < 1.2 * c["H"] / (2 * c["K"][1, 1]) + 0.3   # polynomial in range
        uv = np.full((len(P), 2), np.nan)
        if ok.any():
            p, _ = cv2.projectPoints(Pc[ok].reshape(-1, 1, 3), np.zeros(3), np.zeros(3), c["K"], c["D"])
            uv[ok] = p.reshape(-1, 2)
        return uv

    def _cost_points(self, dist, mask, uv):
        """Per-point cost; points outside the image or on the car body cost self.trunc (no cheating
        by pushing the borders out of view)."""
        H, W = dist.shape
        cost = np.full(len(uv), self.trunc)
        u, v = uv[:, 0], uv[:, 1]
        ok = np.isfinite(u) & (u >= 0) & (u < W - 1) & (v >= 0) & (v < H - 1)
        ui, vi = u[ok].astype(int), v[ok].astype(int)
        good = mask[vi, ui] > 0
        cost[np.flatnonzero(ok)[good]] = np.minimum(dist[vi[good], ui[good]], self.trunc)
        return cost

    def measure(self, dists, pred, min_pts=40):
        """dists: {cam: edge_dist image}. pred: (x, y, yaw) predicted pose.
        Returns dict(x, y, yaw, dlat, dyaw, cost, contrast, n_pts, grid) or None."""
        x, y, yaw = pred
        z, pitch, roll = self.ground.attitude(x, y, yaw)
        if not self.use_slope:
            pitch = roll = 0.0
        z += self.z_offset
        near = self.P[self.tree.query_ball_point([x, y], self.max_range)]
        near = near[np.hypot(near[:, 0] - x, near[:, 1] - y) > self.min_range]
        # 1) fixed point set per camera: visible, on the image, not on the car body, at the prediction
        sel = {}
        for cam, dist in dists.items():
            mask = get_mask(cam, dist.shape)
            uv = self._project(cam, near, *self._cam_pose(cam, x, y, z, yaw, pitch, roll))
            H, W = dist.shape
            u, v = uv[:, 0], uv[:, 1]
            inside = np.isfinite(u) & (u >= 0) & (u < W - 1) & (v >= 0) & (v < H - 1)
            ui, vi = np.where(inside, u, 0).astype(int), np.where(inside, v, 0).astype(int)
            keep = inside & (mask[vi, ui] > 0)
            if keep.any():
                sel[cam] = (near[keep], mask)
        n_pts = sum(len(p) for p, _ in sel.values())
        if n_pts < min_pts:
            return None
        # 2) grid search: shift across the track (body y) and yaw, same points for every candidate
        c, s = np.cos(yaw), np.sin(yaw)
        grid = np.zeros((len(self.lat), len(self.dyaw)))
        for i, dl in enumerate(self.lat):
            px, py = x - s * dl, y + c * dl
            for j, dy in enumerate(self.dyaw):
                tot = 0.0
                for cam, (P, mask) in sel.items():
                    uv = self._project(cam, P, *self._cam_pose(cam, px, py, z, yaw + dy, pitch, roll))
                    tot += self._cost_points(dists[cam], mask, uv).sum()
                grid[i, j] = tot / n_pts + self.prior_px * dl ** 2
        i, j = np.unravel_index(np.argmin(grid), grid.shape)
        prof = grid.min(1)                                                # best cost per lateral shift
        far = np.abs(self.lat - self.lat[i]) >= 0.5
        contrast = float(np.median(prof[far]) - prof[i]) if far.any() else 0.0
        dl, dy = self.lat[i], self.dyaw[j]
        return dict(x=x - s * dl, y=y + c * dl, yaw=yaw + dy, dlat=float(dl), dyaw=float(dy),
                    cost=float(grid[i, j]), contrast=contrast, n_pts=n_pts, grid=grid,
                    at_edge=bool(i in (0, len(self.lat) - 1)))
