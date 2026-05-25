/**
 * kf.c — Linear Kalman Filter implementation.
 *
 * Every matrix operation goes through matrix.h functions — no raw loops
 * here except where we need to read/write individual elements for
 * building small temporary matrices.
 *
 * Temporary buffer naming convention used throughout:
 *   _t    — transposed form
 *   _tmp  — intermediate product, discarded after the expression
 *   Letters A,B,C follow the algebraic derivation in kf.h.
 *
 * Matrix dimensions reminder (n=4, m=1):
 *   x   [4]      P   [4×4]
 *   F   [4×4]    H   [1×4]    Q   [4×4]    R   [1×1]
 *   Ht  [4×1]    FPFt[4×4]    PHt [4×1]
 *   S   [1×1]    K   [4×1]    innovation [1]
 */

#include "kf.h"
#include "matrix.h"

#include <string.h>   /* memcpy, memset */
#include <stdio.h>
#include <math.h>


/* ================================================================== */
/* kf_init                                                             */
/* ================================================================== */

void kf_init(KFState      *kf,
             double        dt,
             SigmaProcess  sp,
             double        sigma_camera,
             const double *x0,
             const double *P0_diag)
{
    double dt2 = dt * dt;
    double dt3 = dt * dt * dt;

    /* ---- Zero the whole struct first ---- */
    memset(kf, 0, sizeof(KFState));

    /* ---- F(dt) — constant curvature-rate clothoid model ---- */
    /*
     * Row-major layout for a 4×4 matrix:
     *
     *   [ 1   dt   dt²/2   dt³/6 ]
     *   [ 0    1     dt    dt²/2 ]
     *   [ 0    0      1      dt  ]
     *   [ 0    0      0       1  ]
     */
    kf->F[0*4+0] = 1.0;  kf->F[0*4+1] = dt;   kf->F[0*4+2] = 0.5*dt2;  kf->F[0*4+3] = dt3/6.0;
    kf->F[1*4+0] = 0.0;  kf->F[1*4+1] = 1.0;  kf->F[1*4+2] = dt;       kf->F[1*4+3] = 0.5*dt2;
    kf->F[2*4+0] = 0.0;  kf->F[2*4+1] = 0.0;  kf->F[2*4+2] = 1.0;      kf->F[2*4+3] = dt;
    kf->F[3*4+0] = 0.0;  kf->F[3*4+1] = 0.0;  kf->F[3*4+2] = 0.0;      kf->F[3*4+3] = 1.0;

    /* ---- H — camera observes C0 only ---- */
    kf->H[0] = 1.0;   /* [1, 0, 0, 0] */

    /* ---- Q — diagonal process noise ---- */
    kf->Q[0*4+0] = sp.C0 * sp.C0;
    kf->Q[1*4+1] = sp.C1 * sp.C1;
    kf->Q[2*4+2] = sp.C2 * sp.C2;
    kf->Q[3*4+3] = sp.C3 * sp.C3;

    /* ---- R — scalar camera measurement noise ---- */
    kf->R[0] = sigma_camera * sigma_camera;

    /* ---- Initial state ---- */
    if (x0) {
        memcpy(kf->x, x0, KF_N * sizeof(double));
    }
    /* else x is already zero from memset */

    /* ---- Initial covariance ---- */
    if (P0_diag) {
        for (int i = 0; i < KF_N; i++)
            kf->P[i*KF_N + i] = P0_diag[i];
    } else {
        /* Default: 10× the process noise diagonal — loose prior */
        kf->P[0*4+0] = 10.0 * sp.C0 * sp.C0;
        kf->P[1*4+1] = 10.0 * sp.C1 * sp.C1;
        kf->P[2*4+2] = 10.0 * sp.C2 * sp.C2;
        kf->P[3*4+3] = 10.0 * sp.C3 * sp.C3;
    }
}


/* ================================================================== */
/* kf_predict — time update                                            */
/* ================================================================== */
/*
 *   x̂⁻ = F · x̂
 *   P⁻  = F · P · Fᵀ + Q
 */
void kf_predict(KFState *kf)
{
    /* --- x̂⁻ = F · x̂ ---
     *
     * x is [4×1], F is [4×4].
     * We treat x as a [4×1] matrix (K=4, C_cols=1).
     */
    double x_new[KF_N];
    mat_mul(kf->F, kf->x, x_new, KF_N, KF_N, 1);
    memcpy(kf->x, x_new, KF_N * sizeof(double));

    /* --- P⁻ = F·P·Fᵀ + Q ---
     *
     * Step 1: tmp1 = F·P          [4×4] = [4×4]×[4×4]
     * Step 2: Ft   = Fᵀ           [4×4]
     * Step 3: FPFt = tmp1·Ft      [4×4] = [4×4]×[4×4]
     * Step 4: P    = FPFt + Q     [4×4]
     */
    double Ft  [KF_N * KF_N];
    double tmp1[KF_N * KF_N];
    double FPFt[KF_N * KF_N];

    mat_transpose(kf->F, Ft,   KF_N, KF_N);
    mat_mul(kf->F, kf->P, tmp1, KF_N, KF_N, KF_N);
    mat_mul(tmp1,  Ft,    FPFt, KF_N, KF_N, KF_N);
    mat_add(FPFt,  kf->Q, kf->P, KF_N, KF_N);
}


/* ================================================================== */
/* kf_update — measurement update                                      */
/* ================================================================== */
/*
 *   y  = z − H·x̂⁻                  [1]
 *   S  = H·P⁻·Hᵀ + R               [1×1]
 *   K  = P⁻·Hᵀ·S⁻¹                 [4×1]
 *   x̂  = x̂⁻ + K·y                  [4]
 *   P  = (I−K·H)·P⁻·(I−K·H)ᵀ + K·R·Kᵀ
 *
 * Dimension legend   n=4  m=1:
 *   Ht     [n×m] = [4×1]   H transposed
 *   PHt    [n×m] = [4×1]   P·Hᵀ
 *   HPHt   [m×m] = [1×1]   H·P·Hᵀ
 *   S      [m×m] = [1×1]   innovation covariance
 *   Sinv   [m×m] = [1×1]   S⁻¹
 *   K      [n×m] = [4×1]   Kalman gain
 *   KH     [n×n] = [4×4]   K·H
 *   IKH    [n×n] = [4×4]   I − K·H
 *   Kt     [m×n] = [1×4]   Kᵀ
 */
int kf_update(KFState *kf, const double *z)
{
    /* ---- y = z − H·x̂⁻ ---- */
    double Hx[KF_M];
    mat_mul(kf->H, kf->x, Hx, KF_M, KF_N, 1);
    mat_sub(z, Hx, kf->innovation, KF_M, 1);

    /* ---- Ht = Hᵀ  [4×1] ---- */
    double Ht[KF_N * KF_M];
    mat_transpose(kf->H, Ht, KF_M, KF_N);

    /* ---- PHt = P·Hᵀ  [4×1] ---- */
    double PHt[KF_N * KF_M];
    mat_mul(kf->P, Ht, PHt, KF_N, KF_N, KF_M);

    /* ---- S = H·PHt + R  [1×1] ---- */
    double HPHt[KF_M * KF_M];
    mat_mul(kf->H, PHt, HPHt, KF_M, KF_N, KF_M);
    mat_add(HPHt, kf->R, kf->S, KF_M, KF_M);

    /* ---- Sinv = S⁻¹  [1×1] ---- */
    double Sinv[KF_M * KF_M];
    if (mat_inv(kf->S, Sinv, KF_M) != 0)
        return -1;   /* singular innovation covariance — skip update */

    /* ---- NIS = yᵀ·S⁻¹·y  (scalar) ---- */
    /* For m=1: NIS = y² / S */
    kf->NIS = kf->innovation[0] * kf->innovation[0] * Sinv[0];

    /* ---- K = PHt·Sinv  [4×1] ---- */
    mat_mul(PHt, Sinv, kf->K, KF_N, KF_M, KF_M);

    /* ---- x̂ = x̂⁻ + K·y  [4] ---- */
    double Ky[KF_N];
    mat_mul(kf->K, kf->innovation, Ky, KF_N, KF_M, 1);
    mat_add(kf->x, Ky, kf->x, KF_N, 1);

    /* ---- Joseph-form covariance update ----
     *
     *   IKH = I − K·H          [4×4]
     *   P   = IKH·P·IKHᵀ + K·R·Kᵀ
     *
     * Step by step:
     *   KH       = K·H          [4×4]
     *   IKH      = I − KH       [4×4]
     *   IKHt     = IKHᵀ         [4×4]
     *   tmp1     = IKH·P        [4×4]
     *   lhs      = tmp1·IKHt    [4×4]   ← left term
     *   Kt       = Kᵀ           [1×4]
     *   KRKt_tmp = R·Kt         [1×4]   (broadcast scalar R)
     *   KRKt     = K·KRKt_tmp   [4×4]   ← right term
     *   P        = lhs + KRKt   [4×4]
     */
    double KH  [KF_N * KF_N];
    double IKH [KF_N * KF_N];
    double IKHt[KF_N * KF_N];
    double tmp1[KF_N * KF_N];
    double lhs [KF_N * KF_N];
    double Kt  [KF_M * KF_N];
    double KRKt[KF_N * KF_N];

    mat_mul(kf->K, kf->H, KH, KF_N, KF_M, KF_N);

    mat_eye(IKH, KF_N);
    mat_sub(IKH, KH, IKH, KF_N, KF_N);

    mat_transpose(IKH, IKHt, KF_N, KF_N);
    mat_mul(IKH,  kf->P, tmp1, KF_N, KF_N, KF_N);
    mat_mul(tmp1, IKHt,  lhs,  KF_N, KF_N, KF_N);

    /* K·R·Kᵀ: R is scalar (1×1), so K·R·Kᵀ = R[0] * K·Kᵀ */
    mat_transpose(kf->K, Kt, KF_N, KF_M);
    mat_mul(kf->K, Kt, KRKt, KF_N, KF_M, KF_N);
    mat_scale(KRKt, kf->R[0], KF_N, KF_N);   /* multiply by scalar R */

    mat_add(lhs, KRKt, kf->P, KF_N, KF_N);

    /* Enforce symmetry — guards against floating-point drift */
    mat_symmetrise(kf->P, KF_N);

    return 0;
}


/* ================================================================== */
/* kf_print                                                            */
/* ================================================================== */

void kf_print(const KFState *kf)
{
    printf("KF State:\n");
    printf("  x̂  = [C0=%8.4f m  C1=%8.5f rad  C2=%9.6f 1/m  C3=%10.7f 1/m²]\n",
           kf->x[0], kf->x[1], kf->x[2], kf->x[3]);
    printf("  P diag = [%8.5f  %8.5f  %8.5f  %8.5f]\n",
           kf->P[0*4+0], kf->P[1*4+1], kf->P[2*4+2], kf->P[3*4+3]);
    printf("  NIS    = %.4f   innovation = %.4f m\n",
           kf->NIS, kf->innovation[0]);
}
