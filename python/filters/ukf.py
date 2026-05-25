"""
ukf.py — Unscented Kalman Filter

Uses the Van der Merwe scaled sigma-point transform to propagate the
Gaussian through the nonlinear functions f and h without computing
any Jacobians.

Sigma-point parameters (α, β, κ):
  α  — spread of sigma points around mean        (typically 1e-3 to 1)
  β  — prior knowledge of distribution (β=2 optimal for Gaussian)
  κ  — secondary scaling parameter               (typically 0 or 3-n)

2n+1 sigma points:
  X₀   = x̂
  Xᵢ   = x̂ + (√((n+λ)P))ᵢ     i = 1…n
  Xᵢ₊ₙ = x̂ - (√((n+λ)P))ᵢ     i = 1…n

  where λ = α²(n + κ) - n

Weights:
  Wₘ⁰  = λ/(n+λ)
  Wᶜ⁰  = λ/(n+λ) + (1 - α² + β)
  Wₘⁱ  = Wᶜⁱ = 1/(2(n+λ))    for i=1…2n

"""

import numpy as np
from typing import Callable, Optional
from .base import BaseFilter


class UnscentedKalmanFilter(BaseFilter):
    """
    Unscented Kalman Filter (Van der Merwe scaled sigma-point form).

    Caller supplies:
        ukf.f_func  — nonlinear state transition  f(x, u, dt) -> x_new  [n]
        ukf.h_func  — nonlinear measurement fn    h(x)        -> z_pred  [m]
        ukf.Q       — process noise covariance    [nxn]
        ukf.R       — measurement noise covariance [mxm]
    """

    def __init__(
        self,
        state_dim: int,
        obs_dim: int,
        alpha: float = 1e-3,
        beta: float = 2.0, 
        kappa: float = 0.0,
    ):
        """
        Args:
            state_dim: Dimension n of state vector.
            obs_dim:   Dimension m of observation vector.
            alpha:     Sigma-point spread parameter.
            beta:      Prior distribution parameter (2 for Gaussian).
            kappa:     Secondary scaling (0 or 3-n).
        """
        super().__init__(state_dim, obs_dim)

        self.alpha = alpha
        self.beta  = beta
        self.kappa = kappa

        # Compute lambda and weights once λ = α²(n + κ) - n
        self._lambda = alpha ** 2 * (state_dim + kappa) - state_dim
        self._compute_weights()

        # Nonlinear functions
        self.f_func: Optional[Callable] = None   # f(x, u, dt) -> x_new
        self.h_func: Optional[Callable] = None   # h(x)        -> z_pred

    # ------------------------------------------------------------------
    # Sigma-point weights
    # ------------------------------------------------------------------

    def _compute_weights(self) -> None:
        n      = self.n
        lam    = self._lambda

        # 2n+1 weights for mean and covariance
        #  Wₘⁱ  = Wᶜⁱ = 1/(2(n+λ))    for i=1…2n
        self.Wm = np.full(2 * n + 1, 1.0 / (2.0 * (n + lam))) # ID array of size 2n+1 with value as 1/2(n+lam)
        self.Wc = np.full(2 * n + 1, 1.0 / (2.0 * (n + lam)))

        # overwrite Wm and Wc for zeroth index
        # Wₘ⁰  = λ/(n+λ)
        self.Wm[0] = lam / (n + lam)
        # Wᶜ⁰  = λ/(n+λ) + (1 - α² + β)
        self.Wc[0] = lam / (n + lam) + (1.0 - self.alpha ** 2 + self.beta)

    # ------------------------------------------------------------------
    # Sigma-point generation
    # ------------------------------------------------------------------

    def _sigma_points(self, x: np.ndarray, P: np.ndarray) -> np.ndarray:
        """
        Generate 2n+1 sigma points from mean x and covariance P.

        Uses Cholesky decomposition of (n+λ)·P for numerical stability.

        Returns:
            X: sigma-point matrix  [2n+1 x n]
        """
        n   = self.n
        lam = self._lambda

        # Cholesky of scaled covariance
        try:
            L = np.linalg.cholesky((n + lam) * P)
        except np.linalg.LinAlgError:
            # If P is near-singular, regularise slightly
            L = np.linalg.cholesky((n + lam) * (P + 1e-9 * np.eye(n)))

        X = np.zeros((2 * n + 1, n))
        X[0] = x
        for i in range(n):
            X[i + 1]     = x + L[:, i]
            X[i + 1 + n] = x - L[:, i]
        return X

    # ------------------------------------------------------------------
    # UKF predict
    # ------------------------------------------------------------------

    def predict(self, u: np.ndarray, dt: float) -> np.ndarray:
        """
        Unscented time update.

        1. Generate sigma points from (x̂, P)
        2. Propagate each sigma point through f
        3. Recover predicted mean and covariance:
             x̂⁻ = Σ Wₘⁱ · f(Xᵢ)
             P⁻  = Σ Wᶜⁱ · (f(Xᵢ)-x̂⁻)(f(Xᵢ)-x̂⁻)ᵀ + Q

        Args:
            u:  Control vector [v, ω]
            dt: Sample period (s)

        Returns:
            x_prior [n]
        """
        # Phase 1: Prediction (Time Update)
        X = self._sigma_points(self.x, self.P)

        # Propagate all sigma points through the nonlinear function
        X_pred = np.array([self.f_func(X[i], u, dt) for i in range(2 * self.n + 1)])

        # Predicted mean
        x_prior = np.einsum('i,ij->j', self.Wm, X_pred)

        # Predicted covariance
        diff = X_pred - x_prior
        P_prior = np.einsum('i,ij,ik->jk', self.Wc, diff, diff) + self.Q

        self.x = x_prior
        self.P = P_prior
        self._X_pred = X_pred   # Cache for cross-covariance in update

        return self.x.copy()

    # ------------------------------------------------------------------
    # UKF update
    # ------------------------------------------------------------------

    def update(self, z: np.ndarray) -> np.ndarray:
        """
        Unscented measurement update.

        Using the propagated sigma points X_pred (from predict step):
          z̄   = Σ Wₘⁱ · h(X_predᵢ)
          S   = Σ Wᶜⁱ · (h(Xᵢ)-z̄)(h(Xᵢ)-z̄)ᵀ + R
          Pxz = Σ Wᶜⁱ · (Xᵢ-x̂⁻)(h(Xᵢ)-z̄)ᵀ
          K   = Pxz · S⁻¹
          x̂  = x̂⁻ + K·(z - z̄)
          P   = P⁻ - K·S·Kᵀ

        Args:
            z: Observation vector [m]

        Returns:
            x_post [n]
        """
        # Phase 2: Update (Measurement Correction)
        z = np.atleast_1d(z)
        n_sigma = 2 * self.n + 1

        # Propagate sigma points through measurement function
        Z_pred = np.array([self.h_func(self._X_pred[i]) for i in range(n_sigma)])

        # Predicted measurement mean
        z_bar = np.einsum('i,ij->j', self.Wm, Z_pred)

        # Innovation covariance S
        dz = Z_pred - z_bar
        self.S = np.einsum('i,ij,ik->jk', self.Wc, dz, dz) + self.R

        # Cross-covariance Pxz
        dx = self._X_pred - self.x
        Pxz = np.einsum('i,ij,ik->jk', self.Wc, dx, dz)

        # Kalman gain K = Pxz. Inv(S)
        self.K = Pxz @ np.linalg.inv(self.S)


        # Phase 3: Final State & Covariance update
        # Innovation
        self.innovation = z - z_bar

        # State update
        self.x = self.x + self.K @ self.innovation

        # Covariance update
        self.P = self.P - self.K @ self.S @ self.K.T

        # Ensure symmetry
        self.P = 0.5 * (self.P + self.P.T)

        return self.x.copy()

    # ------------------------------------------------------------------
    # Factory: build lane-tracking UKF
    # ------------------------------------------------------------------

    @classmethod
    def build_lane_filter(
        cls,
        dt: float,
        sigma_process: tuple = (0.05, 0.005, 5e-5, 1e-6),
        sigma_camera: float = 0.05,
        alpha: float = 1e-3,
        beta: float = 2.0,
        kappa: float = 0.0,
    ) -> "UnscentedKalmanFilter":
        """
        Construct a lane-tracking UKF.

        Uses the same nonlinear f and h as the EKF — the difference is
        that the UKF propagates sigma points instead of Jacobians.

        State:       x = [C0, C1, C2, C3]
        Control:     u = [v (m/s), ω (rad/s)]
        Measurement: z = [C0]
        """
        ukf = cls(state_dim=4, obs_dim=1, alpha=alpha, beta=beta, kappa=kappa)

        # --- Same nonlinear transition as EKF (no Jacobian needed) ------
        def f(x: np.ndarray, u: np.ndarray, dt: float) -> np.ndarray:
            C0, C1, C2, C3 = x
            v, omega = u
            C0_new = C0 - v * dt * np.sin(C1)
            C1_new = C1 + C2 * v * dt - omega * dt
            C2_new = C2 + C3 * v * dt
            C3_new = C3
            return np.array([C0_new, C1_new, C2_new, C3_new])

        # --- Linear measurement: camera sees C0 -------------------------
        def h(x: np.ndarray) -> np.ndarray:
            return np.array([x[0]])

        ukf.f_func = f
        ukf.h_func = h

        # Process noise
        q = np.array(sigma_process) ** 2
        ukf.Q = np.diag(q)
        ukf.R = np.array([[sigma_camera ** 2]])
        ukf.P = np.diag(10.0 * q)

        return ukf
