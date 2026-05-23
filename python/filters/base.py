"""
base.py — Abstract base class for all Kalman filter variants.

Every filter (KF, EKF, UKF) inherits from BaseFilter and must implement:
  - predict(*args, **kwargs)
  - update(z, *args, **kwargs)

State convention for lane tracking:
  x = [C0, C1, C2, C3]
  C0: lateral offset to lane marking (m)
  C1: heading angle relative to lane (rad)
  C2: lane curvature (1/m)
  C3: curvature rate (1/m²)
"""

from abc import ABC, abstractmethod
import numpy as np


class BaseFilter(ABC):
    """Abstract Kalman filter base class."""

    def __init__(self, state_dim: int, obs_dim: int):
        """
        Args:
            state_dim: Dimension n of state vector x.
            obs_dim:   Dimension m of observation vector z.
        """
        self.n = state_dim
        self.m = obs_dim

        # State estimate and covariance
        self.x = np.zeros(state_dim)          # x̂ — posterior mean -> x = F.x
        self.P = np.eye(state_dim)             # P  — posterior covariance -> P = F.P.Ft + Q

        # Noise matrices (caller sets these after construction)
        self.Q = np.eye(state_dim)             # Process noise covariance
        self.R = np.eye(obs_dim)               # Measurement noise covariance

        # Diagnostics — filled each cycle
        self.innovation      = np.zeros(obs_dim)   # Innovatio/Residual -> y = z - H·x̂
        self.S               = np.eye(obs_dim)      # Innovation covariance -> S = H.P.Ht + R
        self.K               = np.zeros((state_dim, obs_dim))  # Kalman gain -> K = P.H.inv(S)

    # ------------------------------------------------------------------
    # Interface that sub-classes must implement
    # ------------------------------------------------------------------

    @abstractmethod
    def predict(self, *args, **kwargs) -> np.ndarray:
        """
        Time-update step.  Propagates x̂ and P forward by one timestep.

        Returns:
            x_prior: Predicted state vector x̂⁻  (shape: [n])
        """

    @abstractmethod
    def update(self, z: np.ndarray, *args, **kwargs) -> np.ndarray:
        """
        Measurement-update step.  Fuses observation z into the estimate.

        Args:
            z: Observation vector (shape: [m])

        Returns:
            x_post: Updated (posterior) state vector x̂  (shape: [n])
        """

    # ------------------------------------------------------------------
    # Shared helpers
    # ------------------------------------------------------------------

    def initialize(self, x0: np.ndarray, P0: np.ndarray) -> None:
        """Set initial state estimate and covariance."""
        assert x0.shape == (self.n,), f"Expected x0 shape ({self.n},), got {x0.shape}"
        assert P0.shape == (self.n, self.n), \
            f"Expected P0 shape ({self.n},{self.n}), got {P0.shape}"
        self.x = x0.copy()
        self.P = P0.copy()

    @property
    def NIS(self) -> float:
        """
        Normalised Innovation Squared — filter consistency check.

        NIS should follow a chi²(m) distribution if the filter is consistent.
        Mean(NIS) ≈ m  (observation dimension).
        """
        # calculate the magnitude of innovation scaled by innovation covariance (S)
        return float(self.innovation @ np.linalg.solve(self.S, self.innovation))

    def __repr__(self) -> str:
        return (
            f"{self.__class__.__name__}("
            f"state_dim={self.n}, obs_dim={self.m})\n"
            f"  x̂ = {np.round(self.x, 4)}\n"
            f"  P diag = {np.round(np.diag(self.P), 4)}"
        )
