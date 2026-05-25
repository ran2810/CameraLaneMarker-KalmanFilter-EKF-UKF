/**
 * ukf.h — Unscented Kalman Filter for lane marker C0 estimation.
 *
 * State vector  x = [C0, C1, C2, C3]ᵀ   (n = UKF_N = 4)
 * Observation   z = [C0]                 (m = UKF_M = 1)
 * Control       u = [v, ω]               (speed m/s, yaw-rate rad/s)
 *
 * Uses the Van der Merwe scaled sigma-point transform (2n+1 = 9 points)
 * to propagate the Gaussian through the nonlinear functions f and h
 * WITHOUT computing any Jacobians.
 *
 * -----------------------------------------------------------------------
 * Sigma-point parameters
 * -----------------------------------------------------------------------
 *   α  — spread of sigma points around mean        (1e-3 to 1)
 *   β  — prior knowledge of distribution (2 = optimal for Gaussian)
 *   κ  — secondary scaling parameter               (0 or 3-n)
 *
 *   λ  = α²(n + κ) − n
 *
 * Sigma points (2n+1 = 9):
 *   X[0]     = x̂
 *   X[i]     = x̂ + col_i( √((n+λ)·P) )    i = 1…n
 *   X[i+n]   = x̂ − col_i( √((n+λ)·P) )    i = 1…n
 *
 *   √((n+λ)·P) is the lower Cholesky factor L  s.t.  L·Lᵀ = (n+λ)·P
 *
 * Weights:
 *   Wm[0]   = λ/(n+λ)
 *   Wc[0]   = λ/(n+λ) + (1 − α² + β)
 *   Wm[i]   = Wc[i] = 1/(2(n+λ))    i = 1…2n
 *
 * -----------------------------------------------------------------------
 * Predict step
 * -----------------------------------------------------------------------
 *   1. Generate sigma points X from (x̂, P)
 *   2. Propagate:  X_f[i] = f(X[i], u, dt)    for i = 0…2n
 *   3. Recover mean and covariance:
 *        x̂⁻ = Σ Wm[i] · X_f[i]
 *        P⁻  = Σ Wc[i] · (X_f[i]−x̂⁻)(X_f[i]−x̂⁻)ᵀ + Q
 *
 * -----------------------------------------------------------------------
 * Update step
 * -----------------------------------------------------------------------
 *   4. Propagate sigma points through h:  Z[i] = h(X_f[i])
 *   5. Predicted measurement mean:
 *        z̄ = Σ Wm[i] · Z[i]
 *   6. Innovation covariance:
 *        S  = Σ Wc[i] · (Z[i]−z̄)(Z[i]−z̄)ᵀ + R
 *   7. Cross-covariance:
 *        Pxz = Σ Wc[i] · (X_f[i]−x̂⁻)(Z[i]−z̄)ᵀ
 *   8. Kalman gain:
 *        K = Pxz · S⁻¹
 *   9. State and covariance update:
 *        x̂  = x̂⁻ + K·(z − z̄)
 *        P   = P⁻ − K·S·Kᵀ
 *
 * Storage note: the predicted sigma points X_f are cached inside UKFState
 * between predict and update — the update reuses them directly.
 *
 * -----------------------------------------------------------------------
 * No heap allocation.  All state lives in UKFState on the caller's stack.
 *
 * Typical usage:
 *
 *   UKFState ukf;
 *   ukf_init(&ukf, sp, sigma_camera, x0, P0_diag,
 *            alpha, beta, kappa);
 *
 *   for each frame k:
 *       double u[2] = { speed_k, yaw_rate_k };
 *       ukf_predict(&ukf, u, dt);
 *       if (measurement_valid)
 *           ukf_update(&ukf, z_k);
 *
 *       double C0_hat = ukf.x[0];
 */

#ifndef UKF_H
#define UKF_H

#include "kf.h"   /* reuse SigmaProcess typedef */

/* ------------------------------------------------------------------ */
/* Compile-time dimensions                                             */
/* ------------------------------------------------------------------ */

#define UKF_N      4               /* state dimension            */
#define UKF_M      1               /* observation dimension       */
#define UKF_NSIGMA (2*UKF_N + 1)  /* number of sigma points = 9 */

/* ------------------------------------------------------------------ */
/* UKF state structure — entirely stack-allocated                      */
/* ------------------------------------------------------------------ */

typedef struct {

    /* --- State estimate --- */
    double x[UKF_N];                        /* posterior mean  x̂       */
    double P[UKF_N * UKF_N];               /* posterior covariance  P  */

    /* --- Fixed noise matrices --- */
    double Q[UKF_N * UKF_N];               /* process noise cov [4×4]  */
    double R[UKF_M * UKF_M];               /* meas noise cov    [1×1]  */

    /* --- Sigma-point weights (computed once in ukf_init) --- */
    double Wm[UKF_NSIGMA];                 /* mean weights  [9]        */
    double Wc[UKF_NSIGMA];                 /* cov  weights  [9]        */
    double lambda;                          /* scaling parameter λ      */

    /* --- Cached predicted sigma points (predict → update handoff) --- */
    double X_pred[UKF_NSIGMA * UKF_N];     /* X_f matrix  [9×4]       */

    /* --- Diagnostics (updated each cycle) --- */
    double innovation[UKF_M];              /* y   = z − z̄              */
    double S[UKF_M * UKF_M];              /* innovation covariance     */
    double K[UKF_N * UKF_M];              /* Kalman gain  [4×1]        */
    double NIS;                            /* Normalised Innovation Sq  */

} UKFState;

/* ------------------------------------------------------------------ */
/* Public API                                                          */
/* ------------------------------------------------------------------ */

/**
 * ukf_init — compute sigma weights and initialise state.
 *
 * @param ukf           Pointer to caller-allocated UKFState.
 * @param sp            Process noise std-devs (C0, C1, C2, C3).
 * @param sigma_camera  Camera C0 noise std-dev (m).
 * @param x0            Initial state [UKF_N].  NULL → zeros.
 * @param P0_diag       Diagonal of initial covariance.  NULL → 10·Q diag.
 * @param alpha         Sigma-point spread (e.g. 1e-3).
 * @param beta          Distribution parameter (2.0 for Gaussian).
 * @param kappa         Secondary scaling (0.0 or 3.0 − UKF_N).
 */
void ukf_init(UKFState     *ukf,
              SigmaProcess  sp,
              double        sigma_camera,
              const double *x0,
              const double *P0_diag,
              double        alpha,
              double        beta,
              double        kappa);

/**
 * ukf_predict — sigma-point time update.
 *
 * @param ukf  UKF state (modified in-place).
 * @param u    Control vector [v (m/s), ω (rad/s)].
 * @param dt   Sample period (s).
 *
 * @return  0 on success, -1 if Cholesky of (n+λ)P fails (P non-PD).
 */
int ukf_predict(UKFState *ukf, const double *u, double dt);

/**
 * ukf_update — sigma-point measurement update.
 *
 * @param ukf  UKF state (modified in-place).  ukf_predict must have
 *             been called first this cycle (X_pred must be populated).
 * @param z    Observation [UKF_M].
 *
 * @return  0 on success, -1 if S is singular.
 */
int ukf_update(UKFState *ukf, const double *z);

/**
 * ukf_print — print current state and diagnostics to stdout.
 */
void ukf_print(const UKFState *ukf);

#endif /* UKF_H */
