"""
generate_lane_data.py — Synthetic A9 → A8 Autobahn Interchange Dataset

Simulates a vehicle travelling from the A9 (München–Nürnberg) onto the A8
(München–Salzburg) via the Autobahndreieck München-Süd.

True lane geometry follows the clothoid/Euler spiral model used in road
design.  Camera detections are simulated by adding:
  - Gaussian noise  (σ_C0 = 0.05 m  — typical vision sensor)
  - Occasional outliers  (~4% of frames)
  - Occasional missing detections  (~3% of frames)

Output (LaneDataset):
  t         [T]       timestamps (s)
  speed     [T]       ego vehicle speed (m/s)
  yaw_rate  [T]       ego yaw rate (rad/s)
  C0_L_true [T]       true left  C0 (m)
  C0_R_true [T]       true right C0 (m)
  C1_true   [T]       true heading angle (rad, shared both sides by sign)
  C2_true   [T]       true curvature (1/m)
  meas_L    [T]       noisy camera C0 left  (NaN = dropout)
  meas_R    [T]       noisy camera C0 right (NaN = dropout)

Scenario timeline
-----------------
Phase 1   0–30 s   Straight A9                speed ≈ 120 km/h
Phase 2  30–50 s   Curve into Dreieck         curvature → 0.003 1/m
Phase 3  50–70 s   Transition/connector road  lane width 3.75→3.50 m, speed down
Phase 4  70–90 s   Joined A8, straight        speed ≈ 110 km/h
Phase 5  90–120 s  Mild S-curve on A8         curvature oscillates ±0.002 1/m
"""

import numpy as np
from dataclasses import dataclass, field
from typing import Optional


# ------------------------------------------------------------------
# Data container
# ------------------------------------------------------------------

@dataclass
class LaneDataset:
    """All arrays have length T = number of camera frames."""

    t:          np.ndarray    # timestamps (s)
    speed:      np.ndarray    # ego speed  (m/s)
    yaw_rate:   np.ndarray    # ego yaw rate (rad/s)

    # Ground truth — four-parameter clothoid model per marker
    C0_L_true:  np.ndarray    # left  lateral offset (m)
    C0_R_true:  np.ndarray    # right lateral offset (m)
    C1_true:    np.ndarray    # heading angle relative to lane (rad)
    C2_true:    np.ndarray    # curvature (1/m)
    C3_true:    np.ndarray    # curvature rate (1/m²)

    # Camera detections (NaN = missing frame)
    meas_L:     np.ndarray    # noisy C0 left  (m)
    meas_R:     np.ndarray    # noisy C0 right (m)

    dt: float = 0.04          # sample period  (s)

    def __post_init__(self):
        T = len(self.t)
        for name, arr in self.__dict__.items():
            if isinstance(arr, np.ndarray) and arr.ndim == 1 and len(arr) != T:
                raise ValueError(f"Array '{name}' length {len(arr)} ≠ T={T}")

    @property
    def T(self) -> int:
        return len(self.t)

    def phase_mask(self, phase: int) -> np.ndarray:
        """Boolean mask for each drive phase (1–5)."""
        boundaries = [0, 30, 50, 70, 90, 120]
        t0, t1 = boundaries[phase - 1], boundaries[phase]
        return (self.t >= t0) & (self.t < t1)


# ------------------------------------------------------------------
# Generator
# ------------------------------------------------------------------

def generate_highway_data(
    dt: float = 0.04,
    duration: float = 120.0,
    seed: int = 42,
    sigma_camera: float = 0.05,
    outlier_rate: float = 0.04,
    dropout_rate: float = 0.03,
) -> LaneDataset:
    """
    Generate the synthetic A9→A8 interchange dataset.

    Args:
        dt:           Camera frame period (s). Default 25 Hz.
        duration:     Total scenario duration (s).
        seed:         Random seed for reproducibility.
        sigma_camera: Std-dev of Gaussian measurement noise (m).
        outlier_rate: Fraction of frames with outlier detections.
        dropout_rate: Fraction of frames with missing detections.

    Returns:
        LaneDataset
    """
    rng = np.random.default_rng(seed)
    T   = int(duration / dt)
    t   = np.arange(T) * dt

    # ------------------------------------------------------------------
    # 1.  Speed and yaw-rate profiles
    # ------------------------------------------------------------------

    speed    = np.zeros(T)
    yaw_rate = np.zeros(T)

    for i, ti in enumerate(t):
        if ti < 30:                      # Phase 1: straight A9 at 120 km/h
            speed[i]    = 33.3
            yaw_rate[i] = 0.0

        elif ti < 50:                    # Phase 2: curve into interchange
            alpha       = (ti - 30) / 20.0
            speed[i]    = 33.3 - 5.6 * alpha       # slow to ~100 km/h
            # Curvature 0 → 0.003 1/m  →  yaw_rate = v × κ
            kappa       = 0.003 * alpha
            yaw_rate[i] = speed[i] * kappa

        elif ti < 70:                    # Phase 3: connector road
            alpha       = (ti - 50) / 20.0
            speed[i]    = 27.8 - 2.8 * alpha       # 100→90 km/h
            kappa       = 0.003 * (1.0 - alpha)    # curvature winds down
            yaw_rate[i] = speed[i] * kappa

        elif ti < 90:                    # Phase 4: joined A8, straight
            speed[i]    = 30.6                      # 110 km/h
            yaw_rate[i] = 0.0

        else:                            # Phase 5: S-curve on A8
            theta       = 2 * np.pi * (ti - 90) / 30.0
            speed[i]    = 30.6
            kappa       = 0.002 * np.sin(theta)
            yaw_rate[i] = speed[i] * kappa

    # Add small measurement noise to IMU/odometry signals
    speed    += rng.normal(0, 0.2, T)
    yaw_rate += rng.normal(0, 5e-4, T)

    # ------------------------------------------------------------------
    # 2.  True lane geometry
    # ------------------------------------------------------------------

    # German Autobahn standard: 3.75 m lane width → half-width = 1.875 m
    # Connector/transition road:  3.50 m lane width → half-width = 1.750 m

    HALF_A9  = 1.875    # m
    HALF_A8  = 1.750    # m

    C0_centre = np.zeros(T)   # true lane half-width (positive = left marker)
    C2_true   = np.zeros(T)   # true curvature
    C3_true   = np.zeros(T)   # curvature rate (small)
    C1_true   = np.zeros(T)   # heading angle (accumulated from yaw-rate integral)

    # Integrate heading from yaw rate:  dψ_rel/dt = C2·v - ω  →  if following lane: ≈ 0
    # In ground truth the vehicle follows the lane perfectly (C1 ≈ 0 at steady state),
    # but during transition manoeuvre there is a small, transient heading error.
    heading_error = np.zeros(T)

    for i in range(1, T):
        ti = t[i]
        if 50 <= ti < 65:
            # During transition the vehicle drifts ~2 cm off centreline
            heading_error[i] = 0.005 * np.sin(np.pi * (ti - 50) / 15)
        else:
            heading_error[i] = 0.0

    C1_true = heading_error

    for i, ti in enumerate(t):
        if ti < 30:
            C0_centre[i] = HALF_A9
            C2_true[i]   = 0.0

        elif ti < 50:
            alpha         = (ti - 30) / 20.0
            C0_centre[i]  = HALF_A9
            C2_true[i]    = 0.003 * alpha          # building curvature

        elif ti < 70:
            alpha         = (ti - 50) / 20.0
            # Lane narrows linearly from A9 to A8 standard
            C0_centre[i]  = HALF_A9 + (HALF_A8 - HALF_A9) * alpha
            C2_true[i]    = 0.003 * (1.0 - alpha)

        elif ti < 90:
            C0_centre[i]  = HALF_A8
            C2_true[i]    = 0.0

        else:
            theta         = 2 * np.pi * (ti - 90) / 30.0
            C0_centre[i]  = HALF_A8
            C2_true[i]    = 0.002 * np.sin(theta)

    # Curvature rate C3 ≈ numerical derivative of C2 (small)
    C3_true[1:]  = np.diff(C2_true) / dt
    C3_true[0]   = C3_true[1]

    # True C0: left is positive, right is negative, small perturbation from C1
    C0_L_true = +C0_centre + 0.02 * C1_true
    C0_R_true = -C0_centre + 0.02 * C1_true

    # ------------------------------------------------------------------
    # 3.  Camera measurements
    # ------------------------------------------------------------------

    def make_measurements(C0_true: np.ndarray) -> np.ndarray:
        """Add noise, outliers, and dropouts to a true C0 signal."""
        meas = C0_true.copy()

        # Gaussian noise
        meas += rng.normal(0, sigma_camera, T)

        # Outliers
        outlier_mask = rng.random(T) < outlier_rate
        meas[outlier_mask] += rng.choice([-1, 1], size=outlier_mask.sum()) * \
                               rng.uniform(0.3, 0.8, size=outlier_mask.sum())

        # Dropouts → NaN
        dropout_mask = rng.random(T) < dropout_rate
        meas[dropout_mask] = np.nan

        return meas

    meas_L = make_measurements(C0_L_true)
    meas_R = make_measurements(C0_R_true)

    return LaneDataset(
        t          = t,
        speed      = speed,
        yaw_rate   = yaw_rate,
        C0_L_true  = C0_L_true,
        C0_R_true  = C0_R_true,
        C1_true    = C1_true,
        C2_true    = C2_true,
        C3_true    = C3_true,
        meas_L     = meas_L,
        meas_R     = meas_R,
        dt         = dt,
    )


# ------------------------------------------------------------------
# Quick sanity check when run directly
# ------------------------------------------------------------------

if __name__ == "__main__":
    ds = generate_highway_data()
    print(f"Dataset: T={ds.T} frames, dt={ds.dt}s, duration={ds.t[-1]:.1f}s")
    print(f"C0_L range: [{ds.C0_L_true.min():.3f}, {ds.C0_L_true.max():.3f}] m")
    print(f"C0_R range: [{ds.C0_R_true.min():.3f}, {ds.C0_R_true.max():.3f}] m")
    print(f"C2 range:   [{ds.C2_true.min():.5f}, {ds.C2_true.max():.5f}] 1/m")
    print(f"Dropouts L: {np.isnan(ds.meas_L).sum()} / {ds.T}")
    print(f"Dropouts R: {np.isnan(ds.meas_R).sum()} / {ds.T}")
