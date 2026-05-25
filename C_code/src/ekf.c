/**
 * ekf.c — Extended Kalman Filter implementation.
 *
 * The EKF differs from the KF in exactly one place: the predict step.
 * Instead of  x̂⁻ = F·x̂  and  P⁻ = F·P·Fᵀ + Q,  we compute:
 *
 *   x̂⁻ = f(x̂, u, dt)               — nonlinear propagation of mean
 *   P⁻  = F_jac·P·F_jacᵀ + Q        — covariance via Jacobian at x̂
 *
 * where F_jac = ∂f/∂x  evaluated analytically at the current x̂.
 *
 * The update step is IDENTICAL to the KF because h(x) = C0 = H·x
 * is already linear — the camera measures C0 directly.
 * We therefore share the Joseph-form update logic between ekf_update
 * and kf_update; here it is written out again for self-containment
 * (avoids a cross-module dependency on kf_update internals).
 *
 * Nonlinear model reminder:
 *   f₁ = C0 − v·dt·sin(C1)
 *   f₂ = C1 + C2·v·dt − ω·dt
 *   f₃ = C2 + C3·v·dt
 *   f₄ = C3
 *
 * Jacobian ∂f/∂x at (x̂, u):
 *   row 0: [ 1,  −v·dt·cos(C1),    0,      0    ]
 *   row 1: [ 0,       1,         v·dt,     0    ]
 *   row 2: [ 0,       0,           1,    v·dt   ]
 *   row 3: [ 0,       0,           0,      1    ]
 *
 * Matrix dimensions:  n=4, m=1
 *   x      [4]       P      [4×4]
 *   F_jac  [4×4]     H      [1×4]     Q  [4×4]    R  [1×1]
 *   Ht     [4×1]     PHt    [4×1]     S  [1×1]    K  [4×1]
 */

#include "ekf.h"
#include "matrix.h"

#include <string.h>   /* memset, memcpy */
#include <stdio.h>
#include <math.h>


/* ================================================================== */
/* ekf_init                                                            */
/* ================================================================== */

void ekf_init(EKFState     *ekf,
              SigmaProcess  sp,
              double        sigma_camera,
              const double *x0,
              const double *P0_diag)
{
    memset(ekf, 0, sizeof(EKFState));

    /* H — camera observes C0 only: [1, 0, 0, 0] */
    ekf->H[0] = 1.0;

    /* Q — diagonal process noise */
    ekf->Q[0*4+0] = sp.C0 * sp.C0;
    ekf->Q[1*4+1] = sp.C1 * sp.C1;
    ekf->Q[2*4+2] = sp.C2 * sp.C2;
    ekf->Q[3*4+3] = sp.C3 * sp.C3;

    /* R — scalar camera noise */
    ekf->R[0] = sigma_camera * sigma_camera;

    /* Initial state */
    if (x0)
        memcpy(ekf->x, x0, EKF_N * sizeof(double));

    /* Initial covariance */
    if (P0_diag) {
        for (int i = 0; i < EKF_N; i++)
            ekf->P[i * EKF_N + i] = P0_diag[i];
    } else {
        ekf->P[0*4+0] = 10.0 * sp.C0 * sp.C0;
        ekf->P[1*4+1] = 10.0 * sp.C1 * sp.C1;
        ekf->P[2*4+2] = 10.0 * sp.C2 * sp.C2;
        ekf->P[3*4+3] = 10.0 * sp.C3 * sp.C3;
    }

    /* Seed F_jac with identity (overwritten on first predict call) */
    mat_eye(ekf->F_jac, EKF_N);
}


/* ================================================================== */
/* ekf_predict — nonlinear time update                                 */
/* ================================================================== */
/*
 * Step 1:  Evaluate F_jac = ∂f/∂x  at current x̂, u.
 * Step 2:  Propagate mean through nonlinear f:
 *            x̂⁻ = f(x̂, u, dt)
 * Step 3:  Propagate covariance via linearisation:
 *            P⁻ = F_jac · P · F_jacᵀ + Q
 *
 * NOTE: F_jac is evaluated at x̂ BEFORE the mean is propagated,
 * matching the standard EKF derivation (linearise around current
 * posterior, not predicted prior).
 */
void ekf_predict(EKFState *ekf, const double *u, double dt)
{
    double v     = u[0];
    double omega = u[1];
    double C0    = ekf->x[0];
    double C1    = ekf->x[1];
    double C2    = ekf->x[2];
    double C3    = ekf->x[3];

    /* ---- Step 1: Jacobian ∂f/∂x at current x̂ ----
     *
     *   [1, -v·dt·cos(C1),    0,      0   ]
     *   [0,      1,          v·dt,    0   ]
     *   [0,      0,           1,    v·dt  ]
     *   [0,      0,           0,      1   ]
     *
     * Only the four non-trivial elements need writing;
     * the rest were zeroed by ekf_init / from the previous step.
     */
    mat_zero(ekf->F_jac, EKF_N, EKF_N);
    ekf->F_jac[0*4+0] = 1.0;
    ekf->F_jac[0*4+1] = -v * dt * cos(C1);   /* ∂f₁/∂C1 */
    ekf->F_jac[1*4+1] = 1.0;
    ekf->F_jac[1*4+2] = v * dt;              /* ∂f₂/∂C2 */
    ekf->F_jac[2*4+2] = 1.0;
    ekf->F_jac[2*4+3] = v * dt;              /* ∂f₃/∂C3 */
    ekf->F_jac[3*4+3] = 1.0;

    /* ---- Step 2: nonlinear mean propagation ----
     *
     *   f₁ = C0 − v·dt·sin(C1)
     *   f₂ = C1 + C2·v·dt − ω·dt
     *   f₃ = C2 + C3·v·dt
     *   f₄ = C3
     */
    ekf->x[0] = C0 - v * dt * sin(C1);
    ekf->x[1] = C1 + C2 * v * dt - omega * dt;
    ekf->x[2] = C2 + C3 * v * dt;
    ekf->x[3] = C3;

    /* ---- Step 3: covariance propagation through Jacobian ----
     *
     *   tmp  = F_jac · P             [4×4]
     *   Fjt  = F_jacᵀ               [4×4]
     *   P⁻   = tmp · Fjt + Q        [4×4]
     */
    double Fjt [EKF_N * EKF_N];
    double tmp [EKF_N * EKF_N];
    double FPFt[EKF_N * EKF_N];

    mat_transpose(ekf->F_jac, Fjt, EKF_N, EKF_N);
    mat_mul(ekf->F_jac, ekf->P, tmp,  EKF_N, EKF_N, EKF_N);
    mat_mul(tmp,         Fjt,   FPFt, EKF_N, EKF_N, EKF_N);
    mat_add(FPFt, ekf->Q, ekf->P,    EKF_N, EKF_N);
}


/* ================================================================== */
/* ekf_update — linear measurement update                              */
/* ================================================================== */
/*
 * Identical to kf_update because h(x) = H·x is linear.
 * Fully written out for self-containment; no dependency on kf.c.
 *
 *   y  = z − H·x̂⁻
 *   S  = H·P⁻·Hᵀ + R
 *   K  = P⁻·Hᵀ·S⁻¹
 *   x̂  = x̂⁻ + K·y
 *   P  = (I−KH)·P⁻·(I−KH)ᵀ + K·R·Kᵀ   (Joseph form)
 */
int ekf_update(EKFState *ekf, const double *z)
{
    /* ---- Innovation  y = z − H·x̂⁻ ---- */
    double Hx[EKF_M];
    mat_mul(ekf->H, ekf->x, Hx, EKF_M, EKF_N, 1);
    mat_sub(z, Hx, ekf->innovation, EKF_M, 1);

    /* ---- Ht [4×1] ---- */
    double Ht[EKF_N * EKF_M];
    mat_transpose(ekf->H, Ht, EKF_M, EKF_N);

    /* ---- PHt = P·Hᵀ [4×1] ---- */
    double PHt[EKF_N * EKF_M];
    mat_mul(ekf->P, Ht, PHt, EKF_N, EKF_N, EKF_M);

    /* ---- S = H·PHt + R [1×1] ---- */
    double HPHt[EKF_M * EKF_M];
    mat_mul(ekf->H, PHt, HPHt, EKF_M, EKF_N, EKF_M);
    mat_add(HPHt, ekf->R, ekf->S, EKF_M, EKF_M);

    /* ---- Sinv = S⁻¹ [1×1] ---- */
    double Sinv[EKF_M * EKF_M];
    if (mat_inv(ekf->S, Sinv, EKF_M) != 0)
        return -1;

    /* ---- NIS = y² / S  (scalar, m=1) ---- */
    ekf->NIS = ekf->innovation[0] * ekf->innovation[0] * Sinv[0];

    /* ---- K = PHt·Sinv [4×1] ---- */
    mat_mul(PHt, Sinv, ekf->K, EKF_N, EKF_M, EKF_M);

    /* ---- x̂ = x̂⁻ + K·y ---- */
    double Ky[EKF_N];
    mat_mul(ekf->K, ekf->innovation, Ky, EKF_N, EKF_M, 1);
    mat_add(ekf->x, Ky, ekf->x, EKF_N, 1);

    /* ---- Joseph-form covariance update ----
     *
     *   KH   = K·H                [4×4]
     *   IKH  = I − KH             [4×4]
     *   IKHt = IKHᵀ               [4×4]
     *   P    = IKH·P·IKHᵀ + K·R·Kᵀ
     *
     * R is scalar: K·R·Kᵀ = R[0] · K·Kᵀ
     */
    double KH  [EKF_N * EKF_N];
    double IKH [EKF_N * EKF_N];
    double IKHt[EKF_N * EKF_N];
    double tmp [EKF_N * EKF_N];
    double lhs [EKF_N * EKF_N];
    double Kt  [EKF_M * EKF_N];
    double KRKt[EKF_N * EKF_N];

    mat_mul(ekf->K, ekf->H, KH, EKF_N, EKF_M, EKF_N);
    mat_eye(IKH, EKF_N);
    mat_sub(IKH, KH, IKH, EKF_N, EKF_N);
    mat_transpose(IKH, IKHt, EKF_N, EKF_N);
    mat_mul(IKH,  ekf->P, tmp, EKF_N, EKF_N, EKF_N);
    mat_mul(tmp,  IKHt,   lhs, EKF_N, EKF_N, EKF_N);

    mat_transpose(ekf->K, Kt, EKF_N, EKF_M);
    mat_mul(ekf->K, Kt, KRKt, EKF_N, EKF_M, EKF_N);
    mat_scale(KRKt, ekf->R[0], EKF_N, EKF_N);

    mat_add(lhs, KRKt, ekf->P, EKF_N, EKF_N);
    mat_symmetrise(ekf->P, EKF_N);

    return 0;
}


/* ================================================================== */
/* ekf_print                                                           */
/* ================================================================== */

void ekf_print(const EKFState *ekf)
{
    printf("EKF State:\n");
    printf("  x̂  = [C0=%8.4f m  C1=%8.5f rad  C2=%9.6f 1/m  C3=%10.7f 1/m²]\n",
           ekf->x[0], ekf->x[1], ekf->x[2], ekf->x[3]);
    printf("  P diag = [%8.5f  %8.5f  %8.5f  %8.5f]\n",
           ekf->P[0*4+0], ekf->P[1*4+1], ekf->P[2*4+2], ekf->P[3*4+3]);
    printf("  NIS    = %.4f   innovation = %.4f m\n",
           ekf->NIS, ekf->innovation[0]);
}
