"""
compare_filters.py — Run KF, EKF, and UKF on the same lane dataset and compare.

Usage:
    cd kalman-filters/python/examples
    python compare_filters.py

Outputs:
    - Console: RMSE / MAE table + NIS consistency summary
    - Plots:   C0 comparison, state components, NIS, per-phase RMSE bar chart
"""

import sys
import os
import numpy as np

# Allow running from any CWD
ROOT_PATH = os.path.join(os.path.dirname(__file__), "..", "..")
sys.path.insert(0, ROOT_PATH)

from python.data.generate_lane_data import generate_highway_data, LaneDataset
from python.filters.kf  import KalmanFilter
from python.filters.ekf import ExtendedKalmanFilter
from python.filters.ukf import UnscentedKalmanFilter
from python.utils.metrics import rmse, mae, nis_statistics, print_metrics_table
from python.utils.visualization import (
    plot_c0_comparison,
    plot_state_components,
    plot_nis,
    plot_rmse_per_phase,
    PHASE_LABELS,
)

import matplotlib
matplotlib.use("Agg")   # headless rendering — change to "TkAgg" for interactive


# ==============================================================
# 1. Generate dataset
# ==============================================================

print("Generating A9 → A8 highway interchange dataset …")
ds = generate_highway_data(dt=0.04, duration=120.0, seed=42, sigma_camera=0.05)
print(f"  {ds.T} frames  |  dt={ds.dt}s  |  {ds.T*ds.dt:.0f}s total")
print(f"  Dropouts:  left={np.isnan(ds.meas_L).sum()}  right={np.isnan(ds.meas_R).sum()}")


# ==============================================================
# 2. Filter parameters — shared across all three filters
# ==============================================================

#  Process noise std-devs:  (σ_C0, σ_C1, σ_C2, σ_C3)
SIGMA_PROCESS = (0.04, 0.004, 4e-5, 8e-7)
SIGMA_CAMERA  = 0.05   # m   — camera C0 detection noise


# Tuned values for KF (for C0) -> NIS 0.986 and RMSE 0.07 and MAE 0.03
#SIGMA_PROCESS = (0.07, 0.004, 4e-5, 8e-7)
#SIGMA_CAMERA  = 0.1  

# ==============================================================
# 3. Run filters
# ==============================================================

def run_filter(filt, ds: LaneDataset, side: str = "L"):
    """
    Run a filter over the full dataset for one lane side.

    Returns arrays:
        x0, x1, x2, x3   — state estimates at each step
        nis_vals          — NIS per step
    """
    meas = ds.meas_L if side == "L" else ds.meas_R
    C0_init = ds.C0_L_true[0] if side == "L" else ds.C0_R_true[0]

    # Initialise state: [C0, C1=0, C2=0, C3=0]
    x0_init = np.array([C0_init, 0.0, 0.0, 0.0])
    P0_init = np.diag([0.1, 0.01, 1e-4, 1e-6])
    filt.initialize(x0_init, P0_init)

    T = ds.T
    out_x = np.full((T, 4), np.nan)
    out_nis = np.full(T, np.nan)

    for k in range(T):
        u  = np.array([ds.speed[k], ds.yaw_rate[k]])
        dt = ds.dt

        # --- Predict ---
        if isinstance(filt, KalmanFilter):
            filt.predict()
        else:
            filt.predict(u=u, dt=dt)

        # --- Update (skip if measurement missing) ---
        z = meas[k]
        if not np.isnan(z):
            filt.update(np.array([z]))
            out_nis[k] = filt.NIS

        out_x[k] = filt.x

    return out_x, out_nis


print("\nRunning filters …")

# ------------- KF -------------
kf_L = KalmanFilter.build_lane_filter(ds.dt, SIGMA_PROCESS, SIGMA_CAMERA)
kf_R = KalmanFilter.build_lane_filter(ds.dt, SIGMA_PROCESS, SIGMA_CAMERA)
kf_out_L, kf_nis_L = run_filter(kf_L, ds, "L")
kf_out_R, kf_nis_R = run_filter(kf_R, ds, "R")
print("  KF  done")

# ------------- EKF -------------
ekf_L = ExtendedKalmanFilter.build_lane_filter(ds.dt, SIGMA_PROCESS, SIGMA_CAMERA)
ekf_R = ExtendedKalmanFilter.build_lane_filter(ds.dt, SIGMA_PROCESS, SIGMA_CAMERA)
ekf_out_L, ekf_nis_L = run_filter(ekf_L, ds, "L")
ekf_out_R, ekf_nis_R = run_filter(ekf_R, ds, "R")
print("  EKF done")

# ------------- UKF -------------
ukf_L = UnscentedKalmanFilter.build_lane_filter(ds.dt, SIGMA_PROCESS, SIGMA_CAMERA,
                                                  alpha=1e-3, beta=2.0, kappa=0.0)
ukf_R = UnscentedKalmanFilter.build_lane_filter(ds.dt, SIGMA_PROCESS, SIGMA_CAMERA,
                                                  alpha=1e-3, beta=2.0, kappa=0.0)
ukf_out_L, ukf_nis_L = run_filter(ukf_L, ds, "L")
ukf_out_R, ukf_nis_R = run_filter(ukf_R, ds, "R")
print("  UKF done")


# ==============================================================
# 4. Metrics
# ==============================================================

def valid_nis(arr):
    return arr[~np.isnan(arr)]

results = {
    "KF":  {"rmse_L": rmse(kf_out_L[:, 0],  ds.C0_L_true),
             "rmse_R": rmse(kf_out_R[:, 0],  ds.C0_R_true),
             "mae_L":  mae(kf_out_L[:, 0],   ds.C0_L_true),
             "mae_R":  mae(kf_out_R[:, 0],   ds.C0_R_true)},
    "EKF": {"rmse_L": rmse(ekf_out_L[:, 0], ds.C0_L_true),
             "rmse_R": rmse(ekf_out_R[:, 0], ds.C0_R_true),
             "mae_L":  mae(ekf_out_L[:, 0],  ds.C0_L_true),
             "mae_R":  mae(ekf_out_R[:, 0],  ds.C0_R_true)},
    "UKF": {"rmse_L": rmse(ukf_out_L[:, 0], ds.C0_L_true),
             "rmse_R": rmse(ukf_out_R[:, 0], ds.C0_R_true),
             "mae_L":  mae(ukf_out_L[:, 0],  ds.C0_L_true),
             "mae_R":  mae(ukf_out_R[:, 0],  ds.C0_R_true)},
}
print_metrics_table(results)

# NIS consistency. 
# NIS > 1 filter is overconfident -> leads to decoupling
print("\nNIS Consistency (should have mean ≈ 1 for m=1):")
for name, nis_arr in [("KF",  kf_nis_L), ("EKF", ekf_nis_L), ("UKF", ukf_nis_L)]:
    stats = nis_statistics(valid_nis(nis_arr), obs_dim=1)
    flag  = "consistent" if stats["consistent"] else "INCONSISTENT"
    print(f"  {name:<4} mean_NIS={stats['mean_NIS']:.3f}  "
          f"(low={stats['ci_lo']:.1f}, " f"high={stats['ci_hi']:.1f}, "
          f"expected={stats['expected']:.1f}) --> model is {flag}")

# Per-phase RMSE
print("\nPer-phase RMSE — C0_L:")
phase_rmse = {name: {} for name in results}
for ph in range(1, 6):
    mask = ds.phase_mask(ph)
    for name, arr in [("KF", kf_out_L), ("EKF", ekf_out_L), ("UKF", ukf_out_L)]:
        phase_rmse[name][ph] = rmse(arr[mask, 0], ds.C0_L_true[mask])
    print(f"  Phase {ph}  KF={phase_rmse['KF'][ph]:.4f}  "
          f"EKF={phase_rmse['EKF'][ph]:.4f}  UKF={phase_rmse['UKF'][ph]:.4f}  m")


# ==============================================================
# 5. Plots
# ==============================================================

estimates = {
    "KF":  {"C0_L": kf_out_L[:, 0],  "C0_R": kf_out_R[:, 0],
             "C1":   kf_out_L[:, 1],  "C2":   kf_out_L[:, 2], "C3": kf_out_L[:, 3]},
    "EKF": {"C0_L": ekf_out_L[:, 0], "C0_R": ekf_out_R[:, 0],
             "C1":   ekf_out_L[:, 1], "C2":   ekf_out_L[:, 2], "C3": ekf_out_L[:, 3]},
    "UKF": {"C0_L": ukf_out_L[:, 0], "C0_R": ukf_out_R[:, 0],
             "C1":   ukf_out_L[:, 1], "C2":   ukf_out_L[:, 2], "C3": ukf_out_L[:, 3]},
}

os.makedirs("plots", exist_ok=True)

fig1 = plot_c0_comparison(
    ds.t, ds.C0_L_true, ds.C0_R_true,
    ds.meas_L, ds.meas_R,
    estimates,
    title="C0 Estimation — A9→A8 Autobahn Interchange (KF / EKF / UKF)",
    save_path="plots/c0_comparison.png",
)

fig2 = plot_state_components(
    ds.t, ds.C1_true, ds.C2_true, ds.C3_true,
    estimates,
    save_path="plots/state_components.png",
)

fig3 = plot_nis(
    ds.t,
    nis_dict={"KF": kf_nis_L, "EKF": ekf_nis_L, "UKF": ukf_nis_L},
    obs_dim=1,
    save_path="plots/nis.png",
)

fig4 = plot_rmse_per_phase(
    phase_rmse,
    phase_labels=[f"Ph {i}" for i in range(1, 6)],
    save_path="plots/rmse_per_phase.png",
)

print("\nPlots saved to ./plots/")
