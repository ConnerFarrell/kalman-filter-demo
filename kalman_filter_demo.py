# -*- coding: utf-8 -*-

"""
kalman_filter_demo.py

Single-file demo for multi-sensor state-space simulation and Kalman filtering.

Features
- Simulates a 2D constant-velocity (CV) target with optional slow turn.
- Sensors:
  * GPS-like (x, y) position (linear)
  * IMU-like acceleration with additive bias (used as control input)
  * Radar (range, bearing) — nonlinear (EKF/UKF)
- Filters implemented from scratch:
  * KF  : linear predict/update (GPS). IMU used as control in predict.
  * EKF : linear predict + nonlinear radar update (Jacobians)
  * UKF : sigma-point predict + nonlinear radar update
- Multi-sensor fusion: sequential updates in order IMU(control) → GPS → Radar.
- Saves CSVs, PNG plots, and summary JSON with RMSE.
- Optionally compares to filterpy/pykalman (if installed) for a GPS-only KF baseline.

Quick start
  python kalman_filter_demo.py --model kf --steps 300 --dt 0.1 --plot
  python kalman_filter_demo.py --model ekf --radar --gps --steps 600 --turn_rate 0.01
  python kalman_filter_demo.py --model ukf --radar --compare_lib --no_plot

This file purposefully avoids nonstandard dependencies. If SciPy is installed,
we may use block_diag; otherwise we build small blocks manually.
"""
from __future__ import annotations


import json
import math
import random
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Tuple, Optional

import numpy as np
import matplotlib.pyplot as plt

try:
    from scipy.linalg import block_diag  # type: ignore
except Exception:  # pragma: no cover
    block_diag = None  # fallback below

# ----------------------------- Utilities -----------------------------

def _block_diag_fallback(*mats: np.ndarray) -> np.ndarray:
    if block_diag is not None:
        return block_diag(*mats)
    # Minimal block_diag for small arrays
    shapes = [A.shape for A in mats]
    H = sum(s[0] for s in shapes)
    W = sum(s[1] for s in shapes)
    out = np.zeros((H, W))
    i = 0
    j = 0
    for A in mats:
        h, w = A.shape
        out[i:i+h, j:j+w] = A
        i += h
        j += w
    return out


def wrap_angle(a: float) -> float:
    """Wrap angle to [-pi, pi)."""
    return (a + np.pi) % (2 * np.pi) - np.pi


# ----------------------------- Arguments -----------------------------

@dataclass
class Args:
    model: str = "kf"  # kf | ekf | ukf
    steps: int = 400
    dt: float = 0.1
    seed: int = 0
    save_dir: str = "outputs"
    gps: bool = True
    imu: bool = False
    radar: bool = False
    turn_rate: float = 0.0
    proc_q: float = 0.2
    gps_std: float = 2.0
    imu_std: float = 0.5
    imu_bias_std: float = 0.02
    radar_range_std: float = 3.0
    radar_bearing_std_deg: float = 2.0
    compare_lib: bool = False
    no_plot: bool = False


def make_args() -> Args:
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--model", choices=["kf", "ekf", "ukf"], default="kf")
    p.add_argument("--steps", type=int, default=400)
    p.add_argument("--dt", type=float, default=0.1)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--save_dir", type=str, default="outputs")
    p.add_argument("--gps", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--imu", action=argparse.BooleanOptionalAction, default=False)
    p.add_argument("--radar", action=argparse.BooleanOptionalAction, default=False)
    p.add_argument("--turn_rate", type=float, default=0.0)
    p.add_argument("--proc_q", type=float, default=0.2)
    p.add_argument("--gps_std", type=float, default=2.0)
    p.add_argument("--imu_std", type=float, default=0.5)
    p.add_argument("--imu_bias_std", type=float, default=0.02)
    p.add_argument("--radar_range_std", type=float, default=3.0)
    p.add_argument("--radar_bearing_std_deg", type=float, default=2.0)
    p.add_argument("--compare_lib", action="store_true")
    p.add_argument("--no_plot", action="store_true")
    # Parse known arguments and ignore the rest
    a, _ = p.parse_known_args()
    return Args(**vars(a))


# ----------------------------- Simulation -----------------------------

# State x = [px, py, vx, vy, bax, bay]^T
STATE_DIM = 6


def build_F_Q(dt: float, proc_q: float) -> Tuple[np.ndarray, np.ndarray]:
    """Build CV state transition F and process noise Q.
    Position-velocity use standard CV continuous white noise model.
    Biases (bax,bay) follow random walk with small spectral density.
    """
    F = np.eye(STATE_DIM)
    F[0, 2] = dt
    F[1, 3] = dt

    q = proc_q  # accel spectral density
    dt2 = dt * dt
    dt3 = dt2 * dt
    dt4 = dt2 * dt2

    Q_posvel = np.array(
        [
            [dt4 / 4 * q, 0, dt3 / 2 * q, 0],
            [0, dt4 / 4 * q, 0, dt3 / 2 * q],
            [dt3 / 2 * q, 0, dt2 * q, 0],
            [0, dt3 / 2 * q, 0, dt2 * q],
        ]
    )
    q_b = max(1e-5, 0.01 * q)
    Q_bias = np.array([[q_b * dt, 0], [0, q_b * dt]])

    Q = _block_diag_fallback(Q_posvel, Q_bias)
    return F, Q


def simulate_truth(args: Args) -> Dict[str, np.ndarray]:
    np.random.seed(args.seed)
    random.seed(args.seed)
    T = args.steps
    dt = args.dt

    x = np.zeros(STATE_DIM)
    # Start near origin, with some velocity
    x[0] = 0.0
    x[1] = 0.0
    x[2] = 10.0  # m/s
    x[3] = 0.0
    # biases initial (zero-mean)
    x[4] = 0.0
    x[5] = 0.0

    F, Q = build_F_Q(dt, args.proc_q)

    px = np.zeros(T)
    py = np.zeros(T)
    vx = np.zeros(T)
    vy = np.zeros(T)
    bax = np.zeros(T)
    bay = np.zeros(T)
    ax_true = np.zeros(T)
    ay_true = np.zeros(T)

    omega = args.turn_rate  # rad/s (small)

    for k in range(T):
        px[k], py[k], vx[k], vy[k], bax[k], bay[k] = x
        # Apply optional coordinated turn to the velocity before integrating
        if abs(omega) > 0:
            v = np.array([x[2], x[3]])
            R = np.array(
                [
                    [math.cos(omega * dt), -math.sin(omega * dt)],
                    [math.sin(omega * dt), math.cos(omega * dt)],
                ]
            )
            v2 = R @ v
            a_vec = (v2 - v) / dt
        else:
            a_vec = np.zeros(2)
        # Record true accel (from the maneuver model + process noise mean zero)
        ax_true[k], ay_true[k] = a_vec

        # Propagate with CV model + small process noise
        x = F @ x
        # Inject process noise using Q^(1/2) * n
        w = np.random.multivariate_normal(mean=np.zeros(STATE_DIM), cov=Q)
        x = x + w
        # Re-apply turn by rotating v-components only (mean effect)
        if abs(omega) > 0:
            v = np.array([x[2], x[3]])
            R = np.array(
                [
                    [math.cos(omega * dt), -math.sin(omega * dt)],
                    [math.sin(omega * dt), math.cos(omega * dt)],
                ]
            )
            x[2:4] = R @ v

    t = np.arange(T) * dt
    return {
        "t": t,
        "px": px,
        "py": py,
        "vx": vx,
        "vy": vy,
        "bax": bax,
        "bay": bay,
        "ax_true": ax_true,
        "ay_true": ay_true,
    }


def simulate_measurements(truth: Dict[str, np.ndarray], args: Args) -> Dict[str, np.ndarray]:
    np.random.seed(args.seed + 1)
    T = args.steps
    meas = {}

    drop_prob = 0.05  # 5% chance a sensor misses a reading

    if args.gps:
        gps = np.full((T, 2), np.nan)
        for k in range(T):
            if random.random() < drop_prob:
                continue
            n = np.random.randn(2) * args.gps_std
            gps[k, 0] = truth["px"][k] + n[0]
            gps[k, 1] = truth["py"][k] + n[1]
        meas["gps"] = gps

    if args.imu:
        # Simulated specific force + bias (discrete approx): a_meas = a_true + b + noise
        imu = np.full((T, 2), np.nan)
        for k in range(T):
            if random.random() < drop_prob:
                continue
            ax = truth["ax_true"][k] + truth["bax"][k]
            ay = truth["ay_true"][k] + truth["bay"][k]
            n = np.random.randn(2) * args.imu_std
            imu[k, 0] = ax + n[0]
            imu[k, 1] = ay + n[1]
        meas["imu"] = imu

    if args.radar:
        radar = np.full((T, 2), np.nan)
        rb_std = math.radians(args.radar_bearing_std_deg)
        for k in range(T):
            if random.random() < drop_prob:
                continue
            px, py = truth["px"][k], truth["py"][k]
            rng = math.hypot(px, py) + np.random.randn() * args.radar_range_std
            bearing = math.atan2(py, px) + np.random.randn() * rb_std
            radar[k, 0] = rng
            radar[k, 1] = bearing
        meas["radar"] = radar

    return meas


# ----------------------------- Linear KF -----------------------------

def kf_predict(x: np.ndarray, P: np.ndarray, F: np.ndarray, Q: np.ndarray,
               u: Optional[np.ndarray] = None, B: Optional[np.ndarray] = None) -> Tuple[np.ndarray, np.ndarray]:
    if u is not None and B is not None:
        x = F @ x + B @ u
    else:
        x = F @ x
    P = F @ P @ F.T + Q
    return x, P


def kf_update(x: np.ndarray, P: np.ndarray, z: np.ndarray, H: np.ndarray, R: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    y = z - H @ x
    S = H @ P @ H.T + R
    K = P @ H.T @ np.linalg.inv(S)
    x = x + K @ y
    I = np.eye(x.shape[0])
    P = (I - K @ H) @ P
    return x, P, K, y


# ----------------------------- Radar (EKF) -----------------------------

def h_radar(x: np.ndarray) -> np.ndarray:
    px, py = x[0], x[1]
    rng = math.hypot(px, py)
    bearing = math.atan2(py, px)
    return np.array([rng, bearing])


def H_jac_radar(x: np.ndarray) -> np.ndarray:
    px, py = x[0], x[1]
    r2 = px * px + py * py
    r = math.sqrt(max(r2, 1e-9))
    H = np.zeros((2, STATE_DIM))
    # dr/dx, dr/dy
    H[0, 0] = px / r
    H[0, 1] = py / r
    # dtheta/dx, dtheta/dy
    H[1, 0] = -py / r2
    H[1, 1] = px / r2
    return H


# ----------------------------- UKF essentials -----------------------------

def sigma_points(x: np.ndarray, P: np.ndarray, alpha: float, beta: float, kappa: float) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    n = x.size
    lam = alpha ** 2 * (n + kappa) - n
    U = np.linalg.cholesky((n + lam) * P)
    sigmas = np.zeros((2 * n + 1, n))
    sigmas[0] = x
    for i in range(n):
        sigmas[i + 1] = x + U[i]
        sigmas[n + i + 1] = x - U[i]
    w_m = np.full(2 * n + 1, 1 / (2 * (n + lam)))
    w_c = np.full(2 * n + 1, 1 / (2 * (n + lam)))
    w_m[0] = lam / (n + lam)
    w_c[0] = lam / (n + lam) + (1 - alpha ** 2 + beta)
    return sigmas, w_m, w_c


def unscented_predict(x: np.ndarray, P: np.ndarray, F: np.ndarray, Q: np.ndarray, dt: float,
                      w_m: np.ndarray, w_c: np.ndarray,
                      u: Optional[np.ndarray] = None, B: Optional[np.ndarray] = None) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    n = x.size
    sigmas, w_m2, w_c2 = sigma_points(x, P, alpha=1e-3, beta=2.0, kappa=0.0)
    # propagate through linear dynamics (with optional control)
    for i in range(sigmas.shape[0]):
        if u is not None and B is not None:
            sigmas[i] = F @ sigmas[i] + B @ u
        else:
            sigmas[i] = F @ sigmas[i]
    x_pred = np.sum(w_m2[:, None] * sigmas, axis=0)
    P_pred = Q.copy()
    for i in range(sigmas.shape[0]):
        dx = (sigmas[i] - x_pred)
        P_pred += w_c2[i] * np.outer(dx, dx)
    return x_pred, P_pred, sigmas


def unscented_update_radar(x: np.ndarray, P: np.ndarray, R: np.ndarray,
                           sigmas: Optional[np.ndarray] = None) -> Tuple[np.ndarray, np.ndarray]:
    # If sigma points not provided, generate from current x,P
    if sigmas is None:
        sigmas, w_m, w_c = sigma_points(x, P, alpha=1e-3, beta=2.0, kappa=0.0)
    else:
        _, w_m, w_c = sigma_points(x, P, alpha=1e-3, beta=2.0, kappa=0.0)

    # Transform sigma points through radar measurement model
    Z = np.zeros((sigmas.shape[0], 2))
    for i, s in enumerate(sigmas):
        z = h_radar(s)
        Z[i] = z
    # Mean of bearing must respect circular stats
    r_mean = np.sum(w_m * Z[:, 0])
    # circular mean for angles
    c = np.sum(w_m * np.cos(Z[:, 1]))
    s = np.sum(w_m * np.sin(Z[:, 1]))
    theta_mean = math.atan2(s, c)
    z_mean = np.array([r_mean, theta_mean])

    # Innovation covariance S and cross-covariance T
    S = R.copy()
    T = np.zeros((x.size, 2))
    for i in range(sigmas.shape[0]):
        dz = Z[i] - z_mean
        dz[1] = wrap_angle(dz[1])
        dx = sigmas[i] - x
        S += w_c[i] * np.outer(dz, dz)
        T += w_c[i] * np.outer(dx, dz)

    K = T @ np.linalg.inv(S)
    # Here we assume actual measurement will be applied outside this function;
    # this function returns the gain and innovation cov; however for simplicity
    # we leave it as a helper for computing K given z externally. We'll do the
    # final update in run_filter.
    return K, S  # Returning K,S for reuse


# ----------------------------- Run Filtering -----------------------------

def run_filter(truth: Dict[str, np.ndarray], meas: Dict[str, np.ndarray], args: Args) -> Dict[str, np.ndarray]:
    T = args.steps
    dt = args.dt

    F, Q = build_F_Q(dt, args.proc_q)

    # Control model for IMU used in predict: u = measured accel (approx specific force)
    # State update with control affects velocities: v += (a - b) * dt, p integrates via F.
    # That corresponds to B such that B@u adds [0,0, dt, dt, 0, 0] applied to accel components,
    # and subtracts bias contribution via state (handled implicitly since biases are in x).
    B = np.zeros((STATE_DIM, 2))
    B[2, 0] = dt
    B[3, 1] = dt

    # Measurement models
    H_gps = np.zeros((2, STATE_DIM))
    H_gps[0, 0] = 1.0
    H_gps[1, 1] = 1.0
    R_gps = np.diag([args.gps_std ** 2, args.gps_std ** 2])

    rb_std = math.radians(args.radar_bearing_std_deg)
    R_radar = np.diag([args.radar_range_std ** 2, rb_std ** 2])

    # Init estimate
    x = np.zeros(STATE_DIM)
    x[0] = 0.0
    x[1] = 0.0
    x[2] = 8.0
    x[3] = 0.0
    P = np.diag([25.0, 25.0, 10.0, 10.0, 1.0, 1.0])

    est = {
        "px": np.zeros(T),
        "py": np.zeros(T),
        "vx": np.zeros(T),
        "vy": np.zeros(T),
        "bax": np.zeros(T),
        "bay": np.zeros(T),
    }

    for k in range(T):
        # 1) Predict: if IMU measurement available, use as control u = a_meas - expected bias.
        u = None
        if args.imu and "imu" in meas:
            z_imu = meas["imu"][k]
            if np.all(np.isfinite(z_imu)):
                # Use raw a_meas as control; bias will be accounted by the state (subtract implicitly)
                # To avoid double-counting bias, we subtract current bias estimate in u
                u = z_imu - x[4:6]
        if args.model == "ukf":
            x, P, _ = unscented_predict(x, P, F, Q, dt, w_m=None, w_c=None, u=u, B=B)
        else:
            x, P = kf_predict(x, P, F, Q, u=u, B=B)

        # 2) Sequential updates: GPS then Radar
        if args.gps and "gps" in meas:
            z = meas["gps"][k]
            if np.all(np.isfinite(z)):
                if args.model == "ukf":
                    # Linear GPS update can be done with standard KF equations
                    x, P, _, _ = kf_update(x, P, z, H_gps, R_gps)
                else:
                    x, P, _, _ = kf_update(x, P, z, H_gps, R_gps)

        if args.radar and "radar" in meas:
            z = meas["radar"][k]
            if np.all(np.isfinite(z)):
                if args.model == "ekf":
                    z_pred = h_radar(x)
                    H = H_jac_radar(x)
                    y = z - z_pred
                    y[1] = wrap_angle(y[1])
                    S = H @ P @ H.T + R_radar
                    K = P @ H.T @ np.linalg.inv(S)
                    x = x + K @ y
                    P = (np.eye(STATE_DIM) - K @ H) @ P
                elif args.model == "ukf":
                    # Build sigma points around current x,P and compute K with UT
                    sigmas, w_m, w_c = sigma_points(x, P, alpha=1e-3, beta=2.0, kappa=0.0)
                    # Transform through measurement
                    Z = np.zeros((sigmas.shape[0], 2))
                    for i, s in enumerate(sigmas):
                        Z[i] = h_radar(s)
                    r_mean = np.sum(w_m * Z[:, 0])
                    c = np.sum(w_m * np.cos(Z[:, 1]))
                    s = np.sum(w_m * np.sin(Z[:, 1]))
                    theta_mean = math.atan2(s, c)
                    z_mean = np.array([r_mean, theta_mean])
                    S = R_radar.copy()
                    T = np.zeros((STATE_DIM, 2))
                    for i in range(sigmas.shape[0]):
                        dz = Z[i] - z_mean
                        dz[1] = wrap_angle(dz[1])
                        dx = sigmas[i] - x
                        S += w_c[i] * np.outer(dz, dz)
                        T += w_c[i] * np.outer(dx, dz)
                    K = T @ np.linalg.inv(S)
                    innov = z - z_mean
                    innov[1] = wrap_angle(innov[1])
                    x = x + K @ innov
                    P = P - K @ S @ K.T
                else:  # plain KF cannot handle radar (nonlinear); skip safely
                    pass

        est["px"][k], est["py"][k], est["vx"][k], est["vy"][k], est["bax"][k], est["bay"][k] = x

    return est


# ----------------------------- Metrics & Comparison -----------------------------

def compute_metrics(truth: Dict[str, np.ndarray], est: Dict[str, np.ndarray]) -> Dict[str, float]:
    def rmse(a, b):
        return float(np.sqrt(np.nanmean((a - b) ** 2)))

    m = {
        "rmse_px": rmse(truth["px"], est["px"]),
        "rmse_py": rmse(truth["py"], est["py"]),
        "rmse_vx": rmse(truth["vx"], est["vx"]),
        "rmse_vy": rmse(truth["vy"], est["vy"]),
    }
    m["rmse_pos"] = float(np.sqrt(m["rmse_px"] ** 2 + m["rmse_py"] ** 2))
    m["rmse_vel"] = float(np.sqrt(m["rmse_vx"] ** 2 + m["rmse_vy"] ** 2))
    return m


def maybe_compare_with_library(truth: Dict[str, np.ndarray], meas: Dict[str, np.ndarray], args: Args) -> Optional[Dict[str, float]]:
    if not args.compare_lib:
        return None

    gps = meas.get("gps", None)
    if gps is None:
        return None

    # Try filterpy first, then pykalman
    try:
        from filterpy.kalman import KalmanFilter  # type: ignore

        kf = KalmanFilter(dim_x=4, dim_z=2)
        dt = args.dt
        kf.F = np.array([[1, 0, dt, 0], [0, 1, 0, dt], [0, 0, 1, 0], [0, 0, 0, 1]])
        q = args.proc_q
        dt2, dt3, dt4 = dt * dt, dt ** 3, dt ** 4
        q_block = np.array([[dt4 / 4 * q, dt3 / 2 * q], [dt3 / 2 * q, dt2 * q]])
        kf.Q = _block_diag_fallback(q_block, q_block)
        kf.H = np.array([[1, 0, 0, 0], [0, 1, 0, 0]])
        kf.R = np.diag([args.gps_std ** 2, args.gps_std ** 2])
        kf.x = np.array([0, 0, 8, 0], dtype=float)
        kf.P = np.diag([25.0, 25.0, 10.0, 10.0])

        est_p = []
        for k in range(args.steps):
            kf.predict()
            if np.all(np.isfinite(gps[k])):
                kf.update(gps[k])
            est_p.append(kf.x.copy())
        est_p = np.array(est_p)
        rmse_px = float(np.sqrt(np.nanmean((truth["px"] - est_p[:, 0]) ** 2)))
        rmse_py = float(np.sqrt(np.nanmean((truth["py"] - est_p[:, 1]) ** 2)))
        return {"lib_rmse_pos": float(np.hypot(rmse_px, rmse_py))}
    except Exception:
        pass

    try:
        from pykalman import KalmanFilter as PKF  # type: ignore

        dt = args.dt
        F = np.array([[1, 0, dt, 0], [0, 1, 0, dt], [0, 0, 1, 0], [0, 0, 0, 1]])
        q = args.proc_q
        dt2, dt3, dt4 = dt * dt, dt ** 3, dt ** 4
        q_block = np.array([[dt4 / 4 * q, dt3 / 2 * q], [dt3 / 2 * q, dt2 * q]])
        Q = _block_diag_fallback(q_block, q_block)
        H = np.array([[1, 0, 0, 0], [0, 1, 0, 0]])
        R = np.diag([args.gps_std ** 2, args.gps_std ** 2])
        kf = PKF(transition_matrices=F, observation_matrices=H,
                 transition_covariance=Q, observation_covariance=R,
                 initial_state_mean=np.array([0, 0, 8, 0]),
                 initial_state_covariance=np.diag([25.0, 25.0, 10.0, 10.0]))
        # Build observation sequence
        obs = gps.copy()
        # pykalman expects nan-safe arrays; we'll mask invalids
        mask = np.isfinite(obs).all(axis=1)
        obs_seq = obs.copy()
        obs_seq[~mask] = np.nan
        est_p, _ = kf.filter(obs_seq)
        rmse_px = float(np.sqrt(np.nanmean((truth["px"] - est_p[:, 0]) ** 2)))
        rmse_py = float(np.sqrt(np.nanmean((truth["py"] - est_p[:, 1]) ** 2)))
        return {"lib_rmse_pos": float(np.hypot(rmse_px, rmse_py))}
    except Exception:
        return None


# ----------------------------- Saving & Plots -----------------------------

def save_csvs(truth: Dict[str, np.ndarray], meas: Dict[str, np.ndarray], est: Dict[str, np.ndarray], save_dir: str) -> None:
    savep = Path(save_dir)
    savep.mkdir(parents=True, exist_ok=True)

    def _save(fname: str, cols: Dict[str, np.ndarray]):
        arr = np.column_stack([cols[k] for k in cols])
        header = ",".join(cols.keys())
        np.savetxt(savep / fname, arr, delimiter=",", header=header, comments="")

    _save("truth.csv", {"t": truth["t"], "px": truth["px"], "py": truth["py"], "vx": truth["vx"], "vy": truth["vy"]})

    if "gps" in meas:
        gps = meas["gps"]
        _save("meas_gps.csv", {"t": truth["t"], "x": gps[:, 0], "y": gps[:, 1]})
    if "imu" in meas:
        imu = meas["imu"]
        _save("meas_imu.csv", {"t": truth["t"], "ax": imu[:, 0], "ay": imu[:, 1]})
    if "radar" in meas:
        radar = meas["radar"]
        _save("meas_radar.csv", {"t": truth["t"], "range": radar[:, 0], "bearing": radar[:, 1]})

    _save("estimates.csv", {"t": truth["t"], "px_hat": est["px"], "py_hat": est["py"],
                             "vx_hat": est["vx"], "vy_hat": est["vy"],
                             "bax_hat": est["bax"], "bay_hat": est["bay"]})


def save_plots(truth: Dict[str, np.ndarray], meas: Dict[str, np.ndarray], est: Dict[str, np.ndarray],
               metrics: Dict[str, float], compare: Optional[Dict[str, float]], save_dir: str, show: bool = True) -> None:
    savep = Path(save_dir)
    savep.mkdir(parents=True, exist_ok=True)

    t = truth["t"]
    # 1) Trajectory
    plt.figure()
    plt.plot(truth["px"], truth["py"], label="true")
    if "gps" in meas:
        gps = meas["gps"]
        plt.scatter(gps[:, 0], gps[:, 1], s=8, alpha=0.4, label="gps noisy")
    if "radar" in meas:
        radar = meas["radar"]
        # Optionally visualize a few radar rays
        for k in range(0, len(t), max(1, len(t)//50)):
            if np.all(np.isfinite(radar[k])):
                r, th = radar[k]
                plt.plot([0, r * math.cos(th)], [0, r * math.sin(th)], alpha=0.2, lw=0.8, color="gray")
    plt.plot(est["px"], est["py"], label="filtered")
    plt.title("Trajectory")
    plt.xlabel("x [m]")
    plt.ylabel("y [m]")
    plt.axis("equal")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig(savep / "trajectory.png", dpi=180)

    # 2) Position time series
    plt.figure()
    plt.plot(t, truth["px"], label="px true")
    plt.plot(t, est["px"], label="px est")
    if "gps" in meas:
        plt.scatter(t, meas["gps"][:, 0], s=6, alpha=0.3, label="px gps")
    plt.plot(t, truth["py"], label="py true")
    plt.plot(t, est["py"], label="py est")
    if "gps" in meas:
        plt.scatter(t, meas["gps"][:, 1], s=6, alpha=0.3, label="py gps")
    plt.title("Position vs Time")
    plt.xlabel("t [s]")
    plt.ylabel("position [m]")
    plt.grid(True)
    plt.legend(ncol=2)
    plt.tight_layout()
    plt.savefig(savep / "position_time_series.png", dpi=180)

    # 3) Velocity time series
    plt.figure()
    plt.plot(t, truth["vx"], label="vx true")
    plt.plot(t, est["vx"], label="vx est")
    plt.plot(t, truth["vy"], label="vy true")
    plt.plot(t, est["vy"], label="vy est")
    plt.title("Velocity vs Time")
    plt.xlabel("t [s]")
    plt.ylabel("velocity [m/s]")
    plt.grid(True)
    plt.legend(ncol=2)
    plt.tight_layout()
    plt.savefig(savep / "velocity_time_series.png", dpi=180)

    # 4) Rolling RMSE (simple running mean of |error|)
    win = max(5, len(t) // 50)
    err_px = np.abs(truth["px"] - est["px"])
    err_py = np.abs(truth["py"] - est["py"])
    ker = np.ones(win) / win
    roll_px = np.convolve(err_px, ker, mode="same")
    roll_py = np.convolve(err_py, ker, mode="same")

    plt.figure()
    plt.plot(t, roll_px, label="|px error| rolling mean")
    plt.plot(t, roll_py, label="|py error| rolling mean")
    plt.title(f"Errors — RMSE pos≈{metrics['rmse_pos']:.2f} m")
    if compare is not None and "lib_rmse_pos" in compare:
        plt.title(
            f"Errors — RMSE pos≈{metrics['rmse_pos']:.2f} m (lib≈{compare['lib_rmse_pos']:.2f} m)"
        )
    plt.xlabel("t [s]")
    plt.ylabel("rolling |error| [m]")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig(savep / "errors_rmse.png", dpi=180)

    if show:
        plt.show()
    else:
        plt.close('all')


# ----------------------------- Main -----------------------------

def main():
    args = make_args()

    # Reproducibility
    np.random.seed(args.seed)
    random.seed(args.seed)

    # Simulate
    truth = simulate_truth(args)
    meas = simulate_measurements(truth, args)

    # Run filter
    est = run_filter(truth, meas, args)

    # Metrics
    metrics = compute_metrics(truth, est)

    # Optional library comparison
    compare = maybe_compare_with_library(truth, meas, args)

    # Save outputs
    savep = Path(args.save_dir)
    savep.mkdir(parents=True, exist_ok=True)

    with open(savep / "summary.json", "w") as f:
        json.dump({
            "args": vars(args),
            "metrics": metrics,
            "compare": compare,
        }, f, indent=2)

    save_csvs(truth, meas, est, args.save_dir)
    save_plots(truth, meas, est, metrics, compare, args.save_dir, show=not args.no_plot)

    # Minimal console summary
    print("Model:", args.model)
    print("RMSE pos (m):", f"{metrics['rmse_pos']:.3f}")
    print("RMSE vel (m/s):", f"{metrics['rmse_vel']:.3f}")
    if compare is not None and 'lib_rmse_pos' in compare:
        print("Library baseline RMSE pos (m):", f"{compare['lib_rmse_pos']:.3f}")


if __name__ == "__main__":
    main()