/**
 * kf.h — Linear Kalman Filter for lane marker C0 estimation.
 *
 * State vector  x = [C0, C1, C2, C3]ᵀ   (n = KF_N = 4)
 *   C0: lateral offset to lane marking (m)
 *   C1: heading angle relative to lane (rad)
 *   C2: lane curvature (1/m)
 *   C3: curvature rate (1/m²)
 *
 * Observation   z = [C0]                 (m = KF_M = 1)
 *   The camera reports the lateral offset to a lane marking.
 *
 * Process model — constant curvature-rate clothoid kinematic model:
 *
 *   F(dt) = [[1,  dt,  dt²/2,  dt³/6],
 *            [0,   1,     dt,  dt²/2],
 *            [0,   0,      1,     dt],
 *            [0,   0,      0,      1]]
 *
 * Measurement model:
 *   H = [1, 0, 0, 0]
 *
 * Covariance update: Joseph form for numerical stability.
 *   P = (I - K·H)·P⁻·(I - K·H)ᵀ + K·R·Kᵀ
 *
 * All matrices are row-major flat double arrays.
 * No heap allocation — all state lives in KFState on the caller's stack
 * (or in static storage for a bare-metal deployment).
 *
 * Typical usage:
 *
 *   KFState kf;
 *   kf_init(&kf, dt, sigma_process, sigma_camera);
 *
 *   for each frame k:
 *       kf_predict(&kf);
 *       if (measurement_valid)
 *           kf_update(&kf, z_k);
 *
 *       // read estimate
 *       double C0_hat = kf.x[0];
 */

#ifndef KF_H
#define KF_H

#include <math.h>

/* ------------------------------------------------------------------ */
/* Compile-time dimensions                                             */
/* ------------------------------------------------------------------ */

#define KF_N  4   /* state dimension  */
#define KF_M  1   /* obs   dimension  */

/* ------------------------------------------------------------------ */
/* KF state structure — everything on the stack, no pointers          */
/* ------------------------------------------------------------------ */

typedef struct {

    /* --- State estimate --- */
    double x[KF_N];              /* posterior mean  x̂        */
    double P[KF_N * KF_N];       /* posterior covariance  P  */

    /* --- Model matrices (set once by kf_init) --- */
    double F[KF_N * KF_N];       /* state transition         */
    double H[KF_M * KF_N];       /* observation              */
    double Q[KF_N * KF_N];       /* process noise cov        */
    double R[KF_M * KF_M];       /* measurement noise cov    */

    /* --- Diagnostics (updated each cycle) --- */
    double innovation[KF_M];     /* y   = z − H·x̂⁻          */
    double S[KF_M * KF_M];       /* S   = H·P⁻·Hᵀ + R       */
    double K[KF_N * KF_M];       /* Kalman gain              */
    double NIS;                  /* Normalised Innovation Sq  */

} KFState;

/* Convenience typedef for a 4-element sigma tuple */
typedef struct { double C0, C1, C2, C3; } SigmaProcess;

/* ------------------------------------------------------------------ */
/* Public API                                                          */
/* ------------------------------------------------------------------ */

/**
 * kf_init — initialise all model matrices and the initial state.
 *
 * @param kf            Pointer to caller-allocated KFState.
 * @param dt            Camera frame period (s), e.g. 0.04 for 25 Hz.
 * @param sp            Process noise std-devs for (C0, C1, C2, C3).
 * @param sigma_camera  Std-dev of camera C0 measurement noise (m).
 * @param x0            Initial state estimate [KF_N].  Pass NULL for zeros.
 * @param P0_diag       Diagonal of initial covariance [KF_N].
 *                      Pass NULL → uses 10 × Q diagonal.
 */
void kf_init(KFState      *kf,
             double        dt,
             SigmaProcess  sp,
             double        sigma_camera,
             const double *x0,
             const double *P0_diag);

/**
 * kf_predict — time update: propagate x̂ and P forward one step.
 *
 *   x̂⁻ = F · x̂
 *   P⁻  = F · P · Fᵀ + Q
 *
 * @param kf  KF state (modified in-place).
 */
void kf_predict(KFState *kf);

/**
 * kf_update — measurement update: fuse observation z into the estimate.
 *
 *   y  = z − H · x̂⁻
 *   S  = H · P⁻ · Hᵀ + R
 *   K  = P⁻ · Hᵀ · S⁻¹
 *   x̂  = x̂⁻ + K · y
 *   P  = (I−KH)·P⁻·(I−KH)ᵀ + K·R·Kᵀ   (Joseph form)
 *
 * @param kf  KF state (modified in-place).
 * @param z   Observation [KF_M].
 *
 * @return  0 on success, -1 if S is singular (measurement skipped).
 */
int kf_update(KFState *kf, const double *z);

/**
 * kf_print — print current state estimate and diagnostics to stdout.
 */
void kf_print(const KFState *kf);

#endif /* KF_H */
