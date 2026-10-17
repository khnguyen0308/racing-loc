"""EKF for the race car in the MAP frame (works on the track and in the pit lane).

State  X = [x, y, yaw, b_w, k_v]
    x, y  [m]      position of constructor0/base_link in map
    yaw   [rad]    heading
    b_w   [rad/s]  bias of the yaw-rate sensor   (true yaw rate = wz - b_w)
    k_v   [-]      speed scale factor           (true speed   = k_v * v)
Predict  : car state speed v (+ optional body lateral speed vy) and yaw rate wz   (allowed inputs)
Update   : any pose-like camera measurement (x, y, yaw) with a full 3x3 covariance,
           e.g. from border matching: small variance across the track, large along it.
Gating   : Mahalanobis distance d2 = r^T S^-1 r against a chi-square threshold.
"""
import numpy as np


def wrap(a):
    return (a + np.pi) % (2 * np.pi) - np.pi


class CarEKF:
    def __init__(self, x0, y0, yaw0, P0=None,
                 sig_v=0.15, sig_w=0.004, sig_bw=2e-5, sig_kv=2e-5, sig_vy=0.1):
        self.X = np.array([x0, y0, yaw0, 0.0, 1.0])
        self.P = np.diag([0.05**2, 0.05**2, np.radians(0.2)**2, 5e-4**2, 3e-3**2]) if P0 is None else P0
        # continuous-time noise densities (per sqrt(s))
        self.sig_v, self.sig_w, self.sig_bw, self.sig_kv, self.sig_vy = sig_v, sig_w, sig_bw, sig_kv, sig_vy
        self.n_upd = self.n_rej = 0

    def predict(self, v, wz, dt, vy=0.0):
        if dt <= 0:
            return
        x, y, yaw, bw, kv = self.X
        c, s = np.cos(yaw), np.sin(yaw)
        vel = kv * v
        self.X = np.array([x + (vel * c - vy * s) * dt, y + (vel * s + vy * c) * dt,
                           wrap(yaw + (wz - bw) * dt), bw, kv])
        F = np.eye(5)
        F[0, 2] = (-vel * s - vy * c) * dt; F[0, 4] = v * c * dt
        F[1, 2] = (vel * c - vy * s) * dt;  F[1, 4] = v * s * dt
        F[2, 3] = -dt
        G = np.zeros((5, 5)); G[0, 0] = c * dt; G[1, 0] = s * dt; G[0, 1] = -s * dt; G[1, 1] = c * dt
        G[2, 2] = dt; G[3, 3] = 1; G[4, 4] = 1
        Qc = np.diag([self.sig_v**2 / dt, self.sig_vy**2 / dt, self.sig_w**2 / dt,
                      self.sig_bw**2 * dt, self.sig_kv**2 * dt])
        self.P = F @ self.P @ F.T + G @ Qc @ G.T

    def update_pose(self, z, R, gate_chi2=11.34):
        """z = (x, y, yaw) measurement, R 3x3 covariance. gate 11.34 = chi2(3 dof, 99 %)."""
        H = np.zeros((3, 5)); H[0, 0] = H[1, 1] = H[2, 2] = 1
        r = np.array([z[0] - self.X[0], z[1] - self.X[1], wrap(z[2] - self.X[2])])
        S = H @ self.P @ H.T + R
        d2 = float(r @ np.linalg.solve(S, r))
        if gate_chi2 is not None and d2 > gate_chi2:
            self.n_rej += 1
            return False, d2
        K = self.P @ H.T @ np.linalg.inv(S)
        self.X = self.X + K @ r; self.X[2] = wrap(self.X[2])
        I_KH = np.eye(5) - K @ H
        self.P = I_KH @ self.P @ I_KH.T + K @ R @ K.T          # Joseph form (stays symmetric PSD)
        self.n_upd += 1
        return True, d2


def body_cov(yaw, sig_lon, sig_lat, sig_yaw):
    """3x3 covariance of a pose measurement whose error is sig_lon along the heading,
    sig_lat across it (e.g. border matching on a straight: sig_lon large, sig_lat small)."""
    c, s = np.cos(yaw), np.sin(yaw)
    Rot = np.array([[c, -s], [s, c]])
    R = np.zeros((3, 3))
    R[:2, :2] = Rot @ np.diag([sig_lon**2, sig_lat**2]) @ Rot.T
    R[2, 2] = sig_yaw**2
    return R
