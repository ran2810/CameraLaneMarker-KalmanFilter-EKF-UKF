"""
generate_lane_data.py — A9 → A8 Autobahn interchange lane dataset.

Simulates a vehicle travelling from the A9 (München - Nürnberg) onto the A8
(München - Salzburg) via the Autobahndreieck München-Süd.

This module serves two purposes:

  1. **Python API** (imported by the filter examples):
         from python.data.generate_lane_data import generate_highway_data
         ds = generate_highway_data()        # returns LaneDataset
         ds.save_csv("data/lane_data.csv")   # optional: write CSV

  2. **Standalone CSV generator** (called by CMake or manually):
         python generate_lane_data.py                          # → data/lane_data.csv
         python generate_lane_data.py path/to/output.csv
         python generate_lane_data.py --out out.csv --seed 7

Scenario: A9 → A8 Autobahn interchange, Munich region, Germany
----------------------------------------------------------------
  Phase 1   0-30 s   Straight A9               120 km/h, κ = 0
  Phase 2  30-50 s   Curve into Dreieck         κ → 0.003 1/m
  Phase 3  50-70 s   Transition/connector road  lane narrows 3.75→3.50 m
  Phase 4  70-90 s   Joined A8, straight        110 km/h, κ = 0
  Phase 5  90-120 s  Mild S-curve on A8         κ ±0.002 1/m

State model: clothoid polynomial  z(s) = C0 + C1·s + C2·s²/2 + C3·s³/6
  C0  lateral offset to lane marking (m)
  C1  heading angle relative to lane (rad)
  C2  lane curvature (1/m)
  C3  curvature rate (1/m²)

Camera measurements: noisy C0 with outliers and dropout frames (NaN).
"""

import argparse
import os
import sys
import math
from dataclasses import dataclass, field

import numpy as np


# =========================================================================
# Data container
# =========================================================================

CSV_HEADER = (
    "t,speed,yaw_rate,"
    "C0_L_true,C0_R_true,C1_true,C2_true,C3_true,"
    "meas_L,meas_R\n"
)

PHASE_BOUNDS = [0, 30, 50, 70, 90, 120]
PHASE_NAMES  = [
    "Straight A9",
    "Curve → Dreieck",
    "Transition road",
    "Joined A8",
    "S-curve A8",
]


@dataclass
class LaneDataset:
    """
    All arrays have length T = number of camera frames.

    meas_L / meas_R contain NaN on dropout frames.
    """

    t:         np.ndarray   # timestamps (s)                    [T]
    speed:     np.ndarray   # ego speed (m/s)                   [T]
    yaw_rate:  np.ndarray   # ego yaw rate (rad/s)              [T]

    # Ground truth
    C0_L_true: np.ndarray   # true left  lateral offset (m)     [T]
    C0_R_true: np.ndarray   # true right lateral offset (m)     [T]
    C1_true:   np.ndarray   # heading angle relative to lane    [T]
    C2_true:   np.ndarray   # curvature (1/m)                   [T]
    C3_true:   np.ndarray   # curvature rate (1/m²)             [T]

    # Camera detections  (NaN = missing frame)
    meas_L:    np.ndarray   # noisy C0 left  (m)                [T]
    meas_R:    np.ndarray   # noisy C0 right (m)                [T]

    dt: float = 0.04        # sample period (s)

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def T(self) -> int:
        """Number of frames."""
        return len(self.t)

    def phase_mask(self, phase: int) -> np.ndarray:
        """Boolean mask for scenario phase 1–5."""
        t0 = PHASE_BOUNDS[phase - 1]
        t1 = PHASE_BOUNDS[phase]
        return (self.t >= t0) & (self.t < t1)

    def summary(self) -> str:
        """One-line summary string."""
        drop_L = int(np.isnan(self.meas_L).sum())
        drop_R = int(np.isnan(self.meas_R).sum())
        return (
            f"LaneDataset  T={self.T}  dt={self.dt}s  "
            f"duration={self.t[-1]:.0f}s  "
            f"dropouts L={drop_L} R={drop_R}"
        )

    # ------------------------------------------------------------------
    # CSV I/O
    # ------------------------------------------------------------------

    def save_csv(self, path: str) -> None:
        """
        Write the dataset to a CSV file.

        Dropout frames are written as ``nan`` so the C parser
        (sscanf + strcmp) can detect them correctly.

        Args:
            path: Destination file path.  Parent directories are created
                  automatically.
        """
        os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
        with open(path, "w") as f:
            f.write(CSV_HEADER)
            for k in range(self.T):
                mL = self.meas_L[k]
                mR = self.meas_R[k]
                f.write(
                    f"{self.t[k]:.4f},"
                    f"{self.speed[k]:.6f},"
                    f"{self.yaw_rate[k]:.8f},"
                    f"{self.C0_L_true[k]:.6f},"
                    f"{self.C0_R_true[k]:.6f},"
                    f"{self.C1_true[k]:.8f},"
                    f"{self.C2_true[k]:.8f},"
                    f"{self.C3_true[k]:.8f},"
                    f"{'nan' if np.isnan(mL) else f'{mL:.6f}'},"
                    f"{'nan' if np.isnan(mR) else f'{mR:.6f}'}\n"
                )
        print(f"Saved {self.T} frames → {path}")

    @staticmethod
    def load_csv(path: str) -> "LaneDataset":
        """
        Read a CSV previously written by save_csv().

        Useful for loading pre-generated datasets in tests or notebooks.
        """
        data = np.genfromtxt(path, delimiter=",", names=True)
        return LaneDataset(
            t         = data["t"],
            speed     = data["speed"],
            yaw_rate  = data["yaw_rate"],
            C0_L_true = data["C0_L_true"],
            C0_R_true = data["C0_R_true"],
            C1_true   = data["C1_true"],
            C2_true   = data["C2_true"],
            C3_true   = data["C3_true"],
            meas_L    = data["meas_L"],
            meas_R    = data["meas_R"],
        )


# =========================================================================
#Generator
# =========================================================================

def generate_highway_data(
    dt:           float = 0.04,
    duration:     float = 120.0,
    seed:         int   = 42,
    sigma_camera: float = 0.05,
    outlier_rate: float = 0.04,
    dropout_rate: float = 0.03,
) -> LaneDataset:
    """
    Generate the synthetic A9 → A8 interchange dataset.

    Args:
        dt:           Camera frame period (s).  Default 0.04 s = 25 Hz.
        duration:     Total scenario duration (s).
        seed:         NumPy random seed for reproducibility.
        sigma_camera: Std-dev of Gaussian C0 measurement noise (m).
        outlier_rate: Fraction of frames with outlier measurements.
        dropout_rate: Fraction of frames with missing detections.

    Returns:
        LaneDataset with ground truth and noisy measurements.
    """
    rng = np.random.default_rng(seed)
    T   = int(duration / dt)
    t   = np.arange(T) * dt

    # ------------------------------------------------------------------
    # 1.  Ego-motion profile  (speed, yaw_rate)
    # ------------------------------------------------------------------
    speed    = np.zeros(T)
    yaw_rate = np.zeros(T)

    for i, ti in enumerate(t):
        if ti < 30:
            speed[i]    = 33.3
            yaw_rate[i] = 0.0
        elif ti < 50:
            a           = (ti - 30) / 20.0
            speed[i]    = 33.3 - 5.6 * a
            kappa       = 0.003 * a
            yaw_rate[i] = speed[i] * kappa
        elif ti < 70:
            a           = (ti - 50) / 20.0
            speed[i]    = 27.8 - 2.8 * a
            kappa       = 0.003 * (1.0 - a)
            yaw_rate[i] = speed[i] * kappa
        elif ti < 90:
            speed[i]    = 30.6
            yaw_rate[i] = 0.0
        else:
            theta       = 2 * math.pi * (ti - 90) / 30.0
            speed[i]    = 30.6
            kappa       = 0.002 * math.sin(theta)
            yaw_rate[i] = speed[i] * kappa

    # Add small measurement noise to IMU/odometry signals
    speed    += rng.normal(0, 0.2, T)
    yaw_rate += rng.normal(0, 5e-4, T)

    # ------------------------------------------------------------------
    # 2.  True lane geometry
    # ------------------------------------------------------------------
    
    # German Autobahn standard: 3.75 m lane width → half-width = 1.875 m
    # Connector/transition road:  3.50 m lane width → half-width = 1.750 m

    HALF_A9 = 1.875   # m  (3.75 m lane / 2)
    HALF_A8 = 1.750   # m  (3.50 m lane / 2)

    C0_hw  = np.zeros(T)   # lane half-width (unsigned)
    C2_true = np.zeros(T)
    C1_true = np.zeros(T)

    for i, ti in enumerate(t):
        if ti < 30:
            C0_hw[i]   = HALF_A9;  C2_true[i] = 0.0
        elif ti < 50:
            a          = (ti - 30) / 20.0
            C0_hw[i]   = HALF_A9;  C2_true[i] = 0.003 * a
        elif ti < 70:
            a          = (ti - 50) / 20.0
            C0_hw[i]   = HALF_A9 + (HALF_A8 - HALF_A9) * a
            C2_true[i] = 0.003 * (1.0 - a)
        elif ti < 90:
            C0_hw[i]   = HALF_A8;  C2_true[i] = 0.0
        else:
            theta      = 2 * math.pi * (ti - 90) / 30.0
            C0_hw[i]   = HALF_A8
            C2_true[i] = 0.002 * math.sin(theta)

    # Small transient heading error during lane-width transition
    for i, ti in enumerate(t):
        if 50 <= ti < 65:
            C1_true[i] = 0.005 * math.sin(math.pi * (ti - 50) / 15)

    # C3 = d(C2)/dt  (numerical derivative)
    C3_true    = np.zeros(T)
    C3_true[1:]= np.diff(C2_true) / dt
    C3_true[0] = C3_true[1]

    C0_L_true = +C0_hw + 0.02 * C1_true
    C0_R_true = -C0_hw + 0.02 * C1_true

    # ------------------------------------------------------------------
    # 3.  Camera measurements  (noise + outliers + dropouts)
    # ------------------------------------------------------------------
    def make_measurements(C0_true: np.ndarray) -> np.ndarray:
        meas = C0_true + rng.normal(0, sigma_camera, T)

        outlier_mask = rng.random(T) < outlier_rate
        n_out = outlier_mask.sum()
        meas[outlier_mask] += (
            rng.choice([-1, 1], size=n_out)
            * rng.uniform(0.3, 0.8, size=n_out)
        )

        dropout_mask = rng.random(T) < dropout_rate
        meas[dropout_mask] = np.nan

        return meas

    return LaneDataset(
        t         = t,
        speed     = speed,
        yaw_rate  = yaw_rate,
        C0_L_true = C0_L_true,
        C0_R_true = C0_R_true,
        C1_true   = C1_true,
        C2_true   = C2_true,
        C3_true   = C3_true,
        meas_L    = make_measurements(C0_L_true),
        meas_R    = make_measurements(C0_R_true),
        dt        = dt,
    )


# =========================================================================
# CLI — run directly to generate a CSV
# =========================================================================

def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Generate the A9→A8 highway lane dataset as CSV.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument(
        "out",
        nargs="?",
        default=os.path.join(
            os.path.dirname(os.path.abspath(__file__)), "..", "..", "data", "lane_data.csv"
        ),
        help="Output CSV path",
    )
    p.add_argument("--dt",           type=float, default=0.04,  metavar="S",
                   help="Sample period (s)")
    p.add_argument("--duration",     type=float, default=120.0, metavar="S",
                   help="Scenario duration (s)")
    p.add_argument("--seed",         type=int,   default=42,
                   help="Random seed")
    p.add_argument("--sigma-camera", type=float, default=0.05,  metavar="M",
                   help="Camera noise std-dev (m)")
    p.add_argument("--outlier-rate", type=float, default=0.04,  metavar="F",
                   help="Fraction of frames with outliers")
    p.add_argument("--dropout-rate", type=float, default=0.03,  metavar="F",
                   help="Fraction of frames with missing detections")
    return p.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    print(f"Generating dataset  dt={args.dt}s  duration={args.duration}s  seed={args.seed}")
    ds = generate_highway_data(
        dt           = args.dt,
        duration     = args.duration,
        seed         = args.seed,
        sigma_camera = args.sigma_camera,
        outlier_rate = args.outlier_rate,
        dropout_rate = args.dropout_rate,
    )
    ds.save_csv(args.out)
    print(ds.summary())
