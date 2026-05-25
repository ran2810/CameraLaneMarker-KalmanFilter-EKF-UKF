"""
ekf.py — Extended Kalman Filter

Model: x_{k+1} = f(x_k, u_k) + w_k,    w_k ~ N(0, Q)
       z_k     = h(x_k)      + v_k,    v_k ~ N(0, R)

The EKF linearises f and h around the current estimate using their
Jacobians (F_jac = ∂f/∂x, H_jac = ∂h/∂x), then applies the standard
KF equations on those linearised approximations.

Lane tracking nonlinear model (with ego-motion compensation):
  Inputs: u = [v, ω]  (vehicle speed m/s, yaw rate rad/s)

  f₁ = C0 - v·dt·sin(C1)       ← nonlinear: ego lateral drift
  f₂ = C1 + C2·v·dt - ω·dt
  f₃ = C2 + C3·v·dt
  f₄ = C3

  Jacobian F_jac (∂f/∂x evaluated at x̂):
  [[1, -v·dt·cos(C1), 0,      0   ],
   [0,      1,        v·dt,   0   ],
   [0,      0,        1,     v·dt ],
   [0,      0,        0,      1   ]]

  Measurement: h(x) = C0  →  H_jac = [1, 0, 0, 0]

"""

import numpy as np
from typing import Callable, Optional, Tuple
from .base import BaseFilter


class ExtendedKalmanFilter(BaseFilter):
    """
    Extended Kalman Filter.

    The caller supplies:
        ekf.f_func     — state transition  f(x, u, dt) -> x_new   [n]
        ekf.F_jac_func — its Jacobian      F_jac(x, u, dt) -> [nxn]
        ekf.h_func     — measurement fn    h(x)    -> z_pred       [m]
        ekf.H_jac_func — its Jacobian      H_jac(x) -> [mxn]
        ekf.Q          — process noise covariance                  [nxn]
        ekf.R          — measurement noise covariance              [mxm]

    Use build_lane_filter() for the lane-marker application.
    """

    def __init__(self, state_dim: int, obs_dim: int):
        super().__init__(state_dim, obs_dim)

        # Nonlinear functions — must be set before use
        self.f_func:     Optional[Callable] = None  # f(x, u, dt) -> x_new
        self.F_jac_func: Optional[Callable] = None  # F_jac(x, u, dt) -> [n×n]
        self.h_func:     Optional[Callable] = None  # h(x) -> z_pred
        self.H_jac_func: Optional[Callable] = None  # H_jac(x) -> [m×n]

    # ------------------------------------------------------------------
    # EKF predict
    # ------------------------------------------------------------------

    def predict(self, u: np.ndarray, dt: float) -> np.ndarray:
        """
        Nonlinear time update.

        x̂⁻ = f(x̂, u, dt)
        F   = ∂f/∂x |_{x̂, u}          (Jacobian)
        P⁻  = F·P·Fᵀ + Q

        Args:
            u:  Control/input vector (e.g., [v, yaw_rate])
            dt: Sample period (s)

        Returns:
            x_prior [n]
        """
        # Propagate mean through nonlinear function
        self.x = self.f_func(self.x, u, dt)

        # Jacobian at current (just-updated) estimate
        F = self.F_jac_func(self.x, u, dt)

        # Propagate covariance through linearisation -> F.P.Ft + Q
        self.P = F @ self.P @ F.T + self.Q  

        return self.x.copy()

    # ------------------------------------------------------------------
    # EKF update
    # ------------------------------------------------------------------

    def update(self, z: np.ndarray) -> np.ndarray:
        """
        Measurement update using linearised h(x).

        y  = z - h(x̂⁻)
        H  = ∂h/∂x |_{x̂⁻}
        S  = H·P⁻·Hᵀ + R
        K  = P⁻·Hᵀ·S⁻¹
        x̂  = x̂⁻ + K·y
        P  = (I - K·H)·P⁻·(I - K·H)ᵀ + K·R·Kᵀ   (Joseph form)

        Args:
            z: Observation vector [m]

        Returns:
            x_post [n]
        """
        z = np.atleast_1d(z)

        # Predicted measurement and Jacobian
        z_pred = self.h_func(self.x)
        H = self.H_jac_func(self.x)

        # Innovation
        self.innovation = z - z_pred

        # Innovation covariance ->S=  H.P.Ht + R (meas. Noise)
        PHt = self.P @ H.T
        self.S = H @ PHt + self.R

        # Kalman gain K = P.H.Inv(S)
        self.K = PHt @ np.linalg.inv(self.S)

        # State update S = x + K.Inv
        self.x = self.x + self.K @ self.innovation

        # Covariance update — Joseph form
        # Simplified form P = (I-K.H).P -> efficint but only valid for optimal K and Numerical senstive
        # Jacob form P = (I-K.H).P.(I-K.H)t + K.R.Kt
        I_KH = np.eye(self.n) - self.K @ H    # (I-HK)
        self.P = I_KH @ self.P @ I_KH.T + self.K @ self.R @ self.K.T

        return self.x.copy()

    # ------------------------------------------------------------------
    # Factory: build lane-tracking EKF
    # ------------------------------------------------------------------

    @classmethod
    def build_lane_filter(
        cls,
        dt: float,
        sigma_process: tuple = (0.05, 0.005, 5e-5, 1e-6),
        sigma_camera: float = 0.05,
    ) -> "ExtendedKalmanFilter":
        """
        Construct a lane-tracking EKF with ego-motion compensation.

        State:       x = [C0, C1, C2, C3]
        Control:     u = [v (m/s), ω (rad/s)]
        Measurement: z = [C0]

        The nonlinear transition includes sin(C1) — the EKF linearises
        this at each step via ∂f/∂C1 = -v·dt·cos(C1).
        """
        ekf = cls(state_dim=4, obs_dim=1)

        # --- Nonlinear state transition -----------------------------------
        def f(x: np.ndarray, u: np.ndarray, dt: float) -> np.ndarray:
            C0, C1, C2, C3 = x
            v, omega = u
            C0_new = C0 - v * dt * np.sin(C1)
            C1_new = C1 + C2 * v * dt - omega * dt
            C2_new = C2 + C3 * v * dt
            C3_new = C3
            return np.array([C0_new, C1_new, C2_new, C3_new])

        # --- Jacobian of f w.r.t. x ---------------------------------------
        #  f₁ = C0_new = C0 - v·dt·sin(C1)      
        #  f₂ = C1_new = C1 + C2·v·dt - ω·dt
        #  f₃ = C2_new = C2 + C3·v·dt
        #  f₄ = C3_new = C3

        # F_jac is partial derivate of Fi wrt C0, C1, C2, C3 for row i
        # ∂f₁/∂C0, ∂f₁/∂C1, ∂f₁/∂C2, ∂f₁/∂C3
        # ∂f₂/∂C0, ∂f₂/∂C1, ∂f₂/∂C2, ∂f₂/∂C3
        # ∂f₃/∂C0, ∂f₃/∂C1, ∂f₃/∂C2, ∂f₃/∂C3
        # ∂f₄/∂C0, ∂f₄/∂C1, ∂f₄/∂C2, ∂f₄/∂C3


        def F_jac(x: np.ndarray, u: np.ndarray, dt: float) -> np.ndarray:
            C0, C1, C2, C3 = x
            v, omega = u
            return np.array([
                [1.0, -v * dt * np.cos(C1), 0.0,      0.0    ],   
                [0.0,       1.0,            v * dt,   0.0    ],  
                [0.0,       0.0,            1.0,      v * dt ],  
                [0.0,       0.0,            0.0,      1.0    ],  
            ])

        # --- Linear measurement model (camera sees C0 only) ---------------
        def h(x: np.ndarray) -> np.ndarray:
            return np.array([x[0]])

        def H_jac(x: np.ndarray) -> np.ndarray:
            return np.array([[1.0, 0.0, 0.0, 0.0]])

        ekf.f_func     = f
        ekf.F_jac_func = F_jac
        ekf.h_func     = h
        ekf.H_jac_func = H_jac

        # Process noise
        q = np.array(sigma_process) ** 2
        ekf.Q = np.diag(q)
        ekf.R = np.array([[sigma_camera ** 2]])
        ekf.P = np.diag(10.0 * q)

        return ekf
