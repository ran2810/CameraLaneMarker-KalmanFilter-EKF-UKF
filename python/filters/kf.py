"""
kf.py — Linear Kalman Filter

Model: x_{k+1} = F·x_k + w_k,    w_k ~ N(0, Q)
       z_k     = H·x_k + v_k,    v_k ~ N(0, R)

For lane tracking the state transition matrix F(dt) encodes a
constant-curvature-rate kinematic model:

   C0_{k+1} = C0_k + C1_k·dt + 0.5·C2_k·dt² + (1/6)·C3_k·dt³
   C1_{k+1} = C1_k + C2_k·dt + 0.5·C3_k·dt²
   C2_{k+1} = C2_k + C3_k·dt
   C3_{k+1} = C3_k

The camera measures only C0, so H = [1, 0, 0, 0].

"""

import numpy as np
from .base import BaseFilter


class KalmanFilter(BaseFilter):
    """
    Standard Linear Kalman Filter.

        kf.F  — state transition matrix          [nxn]
        kf.H  — observation matrix               [mxn]
        kf.Q  — process noise covariance         [nxn]
        kf.R  — measurement noise covariance     [mxm]

    Call initialize(x0, P0) before the first predict/update cycle.
    """

    def __init__(self, state_dim: int, obs_dim: int):
        super().__init__(state_dim, obs_dim)
        self.F = np.eye(state_dim)            # State transition matrix
        self.H = np.zeros((obs_dim, state_dim))  # Observation matrix

    # ------------------------------------------------------------------
    # KF predict  (time update)
    # ------------------------------------------------------------------

    def predict(self) -> np.ndarray:
        """
        Propagate state estimate one step forward.

        x̂(k,k-1) = F · x̂(k-1,k-1)           (state prediction)
        P(k,k-1) = F · P(k-1,k-1) · Ft + Q  (covariance prediction)

        Returns:
            x_prior  [n]
        """
        self.x = self.F @ self.x  # @ is dot product or np.dot()
        self.P = self.F @ self.P @ self.F.T + self.Q
        return self.x.copy()

    # ------------------------------------------------------------------
    # KF update  (measurement update)
    # ------------------------------------------------------------------

    def update(self, z: np.ndarray) -> np.ndarray:
        """
        Incorporate measurement z.

        y  = z - H · x̂⁻                (innovation)
        S  = H · P⁻ · Hᵀ + R           (innovation covariance)
        K  = P⁻ · Hᵀ · S⁻¹             (Kalman gain)
        x̂  = x̂⁻ + K · y                (updated state estimate)
        P  = (I - K·H) · P⁻  · (I - K·H)ᵀ + K·R·Kᵀ   (Joseph form -> numerically stable)

        Args:
            z: Observation vector  [m]

        Returns:
            x_post  [n]
        """
        z = np.atleast_1d(z)

        # Innovation
        self.innovation = z - self.H @ self.x

        # Innovation covariance
        PHt = self.P @ self.H.T
        self.S = self.H @ PHt + self.R

        # Kalman gain
        self.K = PHt @ np.linalg.inv(self.S)

        # State update
        self.x = self.x + self.K @ self.innovation

        # Covariance update — Joseph form for numerical stability
        # Simplified form P = (I-K.H).P -> efficint but only valid for optimal K and Numerical senstive
        # Jacob form P = (I-K.H).P.(I-K.H)t + K.R.Kt -> computational cost but numerical stable and valid for any value of K 
        I_KH = np.eye(self.n) - self.K @ self.H   # (I - K.H)
        self.P = I_KH @ self.P @ I_KH.T + self.K @ self.R @ self.K.T

        return self.x.copy()

    # ------------------------------------------------------------------
    # Factory: build a lane-tracking KF
    # ------------------------------------------------------------------

    @classmethod
    def build_lane_filter(
        cls,
        dt: float,
        sigma_process: tuple = (0.05, 0.005, 5e-5, 1e-6),
        sigma_camera: float = 0.05,
    ) -> "KalmanFilter":
        """
        Construct a ready-to-use lane-tracking KF.

        State:  x = [C0, C1, C2, C3]
        Measurement: z = [C0]

        Args:
            dt:             Sample period (s)
            sigma_process:  Std-devs for (C0, C1, C2, C3) process noise
            sigma_camera:   Std-dev of camera C0 measurement noise (m)

        Returns:
            Initialised KalmanFilter (x0=0, P0=diag(10·sigma_process²))
        """
        kf = cls(state_dim=4, obs_dim=1)

        # State transition — constant curvature-rate kinematic model
        dt2 = dt ** 2
        dt3 = dt ** 3
        kf.F = np.array([
            [1.0, dt,  0.5 * dt2,  dt3 / 6.0],
            [0.0, 1.0, dt,         0.5 * dt2 ],
            [0.0, 0.0, 1.0,        dt        ],
            [0.0, 0.0, 0.0,        1.0       ],
        ])

        # Observation — camera sees C0 only
        kf.H = np.array([[1.0, 0.0, 0.0, 0.0]])

        # Process noise — diagonal, tuned per coefficient
        q = np.array(sigma_process) ** 2
        kf.Q = np.diag(q)

        # Measurement noise
        kf.R = np.array([[sigma_camera ** 2]])

        # Initial covariance — loose prior
        kf.P = np.diag(10.0 * q)

        return kf
