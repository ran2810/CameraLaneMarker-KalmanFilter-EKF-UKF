/**
 * ekf.h — Extended Kalman Filter for lane marker C0 estimation.
 *
 * State vector  x = [C0, C1, C2, C3]ᵀ   (n = EKF_N = 4)
 * Observation   z = [C0]                 (m = EKF_M = 1)
 * Control       u = [v, ω]               (speed m/s, yaw-rate rad/s)
 *
 * Nonlinear process model (ego-motion compensation):
 *
 *   f₁ = C0 − v·dt·sin(C1)     ← nonlinear term; KF uses sin≈C1
 *   f₂ = C1 + C2·v·dt − ω·dt
 *   f₃ = C2 + C3·v·dt
 *   f₄ = C3
 *
 * The EKF linearises f around the current estimate x̂ each step via
 * the analytically computed Jacobian F_jac = ∂f/∂x:
 *
 *   F_jac = [[1, −v·dt·cos(C1), 0,      0    ],
 *            [0,      1,        v·dt,   0    ],
 *            [0,      0,         1,    v·dt  ],
 *            [0,      0,         0,     1    ]]
 *
 * Predict:
 *   x̂⁻ = f(x̂, u, dt)             (propagate through nonlinear model)
 *   P⁻  = F_jac · P · F_jacᵀ + Q  (propagate covariance via Jacobian)
 *
 * Update: identical to linear KF (H is linear — camera sees C0 only):
 *   y  = z − H · x̂⁻
 *   S  = H · P⁻ · Hᵀ + R
 *   K  = P⁻ · Hᵀ · S⁻¹
 *   x̂  = x̂⁻ + K · y
 *   P  = (I−KH)·P⁻·(I−KH)ᵀ + K·R·Kᵀ   (Joseph form)
 *
 * All matrices are row-major flat double arrays.
 * No heap allocation — all state lives in EKFState on the caller's stack.
 *
 * Typical usage:
 *
 *   EKFState ekf;
 *   ekf_init(&ekf, dt, sigma_process, sigma_camera, x0, P0_diag);
 *
 *   for each frame k:
 *       double u[2] = { speed_k, yaw_rate_k };
 *       ekf_predict(&ekf, u, dt);
 *       if (measurement_valid)
 *           ekf_update(&ekf, z_k);
 *
 *       double C0_hat = ekf.x[0];
 */

#ifndef EKF_H
#define EKF_H

#include "kf.h"   /* reuse SigmaProcess typedef — same noise model */

/* ------------------------------------------------------------------ */
/* Compile-time dimensions (shared with KF)                            */
/* ------------------------------------------------------------------ */

#define EKF_N  KF_N   /* 4 */
#define EKF_M  KF_M   /* 1 */

/* ------------------------------------------------------------------ */
/* EKF state structure                                                 */
/* ------------------------------------------------------------------ */

typedef struct {

    /* --- State estimate --- */
    double x[EKF_N];              /* posterior mean  x̂              */
    double P[EKF_N * EKF_N];      /* posterior covariance  P         */

    /* --- Fixed noise matrices (set once by ekf_init) --- */
    double H[EKF_M * EKF_N];      /* observation matrix  [1×4]       */
    double Q[EKF_N * EKF_N];      /* process noise covariance [4×4]  */
    double R[EKF_M * EKF_M];      /* measurement noise cov  [1×1]    */

    /* --- Diagnostics (updated each cycle) --- */
    double innovation[EKF_M];     /* y  = z − H·x̂⁻                  */
    double S[EKF_M * EKF_M];      /* S  = H·P⁻·Hᵀ + R               */
    double K[EKF_N * EKF_M];      /* Kalman gain  [4×1]              */
    double NIS;                   /* Normalised Innovation Squared    */

    /* --- Linearisation (recomputed every predict call) --- */
    double F_jac[EKF_N * EKF_N];  /* ∂f/∂x evaluated at (x̂, u, dt)  */

} EKFState;

/* ------------------------------------------------------------------ */
/* Public API                                                          */
/* ------------------------------------------------------------------ */

/**
 * ekf_init — initialise noise matrices and the initial state estimate.
 *
 * @param ekf           Pointer to caller-allocated EKFState.
 * @param sp            Process noise std-devs for (C0, C1, C2, C3).
 * @param sigma_camera  Std-dev of camera C0 measurement noise (m).
 * @param x0            Initial state [EKF_N].  Pass NULL for zeros.
 * @param P0_diag       Diagonal of initial covariance [EKF_N].
 *                      Pass NULL → uses 10 × Q diagonal.
 */
void ekf_init(EKFState     *ekf,
              SigmaProcess  sp,
              double        sigma_camera,
              const double *x0,
              const double *P0_diag);

/**
 * ekf_predict — nonlinear time update.
 *
 * Propagates state through f(x̂, u, dt) and covariance through F_jac.
 *
 * @param ekf  EKF state (modified in-place).
 * @param u    Control vector [v (m/s), ω (rad/s)].
 * @param dt   Sample period (s).
 */
void ekf_predict(EKFState *ekf, const double *u, double dt);

/**
 * ekf_update — linear measurement update (H is constant).
 *
 * Identical algebra to kf_update; H is [1, 0, 0, 0].
 *
 * @param ekf  EKF state (modified in-place).
 * @param z    Observation [EKF_M].
 *
 * @return  0 on success, -1 if S is singular (measurement skipped).
 */
int ekf_update(EKFState *ekf, const double *z);

/**
 * ekf_print — print current state estimate and diagnostics to stdout.
 */
void ekf_print(const EKFState *ekf);

#endif /* EKF_H */
