/**
 * ukf.c — Unscented Kalman Filter implementation.
 *
 * See ukf.h for the full mathematical derivation.
 *
 * Key design decisions
 * --------------------
 * 1. Sigma-point matrix layout:
 *    X_pred is stored as [UKF_NSIGMA × UKF_N] = [9 × 4], row-major.
 *    Row i  → sigma point i  → X_pred[i*UKF_N + j] for column j.
 *
 * 2. No Jacobian anywhere.  The only "special" linear algebra needed
 *    beyond the KF/EKF is the Cholesky decomposition (mat_chol) for
 *    generating sigma points from (n+λ)·P.
 *
 * 3. Robust Cholesky with adaptive relative regularisation:
 *    With α=1e-3, n+λ ≈ 4e-6.  Adding a *fixed* ε to the diagonal of
 *    (n+λ)·P inflates sigma-point spread by ε/((n+λ)·P[i][i]), which
 *    is 25× for C3 at initialisation — corrupting the covariance.
 *    Instead we try Cholesky unregularised first; on failure we add a
 *    *relative* perturbation that scales with each diagonal element,
 *    matching Python's try-except approach.
 *
 * 4. UKF covariance downdate uses P⁻ − K·S·Kᵀ (standard UKF form).
 *    mat_symmetrise is applied afterwards to prevent drift.
 *
 * 5. h(x) = C0 = x[0] is inlined (no function pointer per sigma point).
 *
 * Notation:
 *   n   = UKF_N = 4    state dim
 *   m   = UKF_M = 1    obs dim
 *   ns  = 2n+1  = 9    sigma count
 */

#include "ukf.h"
#include "matrix.h"

#include <string.h>
#include <stdio.h>
#include <math.h>


/* ================================================================== */
/* Internal: nonlinear state-transition  f(x, u, dt)                  */
/* ================================================================== */
/*
 *   f₁ = C0 − v·dt·sin(C1)
 *   f₂ = C1 + C2·v·dt − ω·dt
 *   f₃ = C2 + C3·v·dt
 *   f₄ = C3
 */
static void f_lane(const double *x, const double *u, double dt, double *xout)
{
    double C0    = x[0];
    double C1    = x[1];
    double C2    = x[2];
    double C3    = x[3];
    double v     = u[0];
    double omega = u[1];

    xout[0] = C0 - v * dt * sin(C1);
    xout[1] = C1 + C2 * v * dt - omega * dt;
    xout[2] = C2 + C3 * v * dt;
    xout[3] = C3;
}

/* ================================================================== */
/* Internal: measurement function  h(x) = C0 = x[0]                  */
/* ================================================================== */
static double h_lane(const double *x)
{
    return x[0];
}

/* ================================================================== */
/* Internal: robust Cholesky — try without regularisation first;      */
/*           on failure add a relative perturbation and retry.         */
/*                                                                     */
/* Matching Python's numpy behaviour:                                  */
/*   try: L = chol(A)                                                 */
/*   except: L = chol(A + 1e-9 * I * max_diag)                       */
/*                                                                     */
/* The relative form prevents the regulariser from dominating small    */
/* diagonal entries (which happens with a fixed absolute ε).           */
/* ================================================================== */
static int robust_chol(const double *A, double *L, size_t N)
{
    /* First attempt — no regularisation */
    if (mat_chol(A, L, N) == 0)
        return 0;

    /* Failed: find largest diagonal for relative scaling */
    double max_diag = 0.0;
    for (size_t i = 0; i < N; i++) {
        double d = fabs(A[i * N + i]);
        if (d > max_diag) max_diag = d;
    }
    if (max_diag < 1e-300) max_diag = 1.0;

    /* Retry with increasing relative perturbations: 1e-9, 1e-7, 1e-5 */
    double eps_levels[] = {1e-9, 1e-7, 1e-5, 1e-3};
    int n_levels = (int)(sizeof(eps_levels) / sizeof(eps_levels[0]));

    double Areg[MATRIX_MAX_DIM * MATRIX_MAX_DIM];
    for (int lvl = 0; lvl < n_levels; lvl++) {
        double eps = eps_levels[lvl] * max_diag;
        mat_copy(A, Areg, N, N);
        for (size_t i = 0; i < N; i++)
            Areg[i * N + i] += eps;
        if (mat_chol(Areg, L, N) == 0)
            return 0;
    }

    return -1;   /* genuinely not positive-definite */
}


/* ================================================================== */
/* ukf_init                                                            */
/* ================================================================== */

void ukf_init(UKFState     *ukf,
              SigmaProcess  sp,
              double        sigma_camera,
              const double *x0,
              const double *P0_diag,
              double        alpha,
              double        beta,
              double        kappa)
{
    memset(ukf, 0, sizeof(UKFState));

    /* ---- λ = α²(n + κ) − n ---- */
    double n      = (double)UKF_N;
    ukf->lambda   = alpha * alpha * (n + kappa) - n;
    double lam    = ukf->lambda;

    /* ---- Sigma-point weights ---- */
    double w_common = 1.0 / (2.0 * (n + lam));

    ukf->Wm[0] = lam / (n + lam);
    ukf->Wc[0] = lam / (n + lam) + (1.0 - alpha * alpha + beta);

    for (int i = 1; i < UKF_NSIGMA; i++) {
        ukf->Wm[i] = w_common;
        ukf->Wc[i] = w_common;
    }

    /* ---- Noise matrices ---- */
    ukf->Q[0*4+0] = sp.C0 * sp.C0;
    ukf->Q[1*4+1] = sp.C1 * sp.C1;
    ukf->Q[2*4+2] = sp.C2 * sp.C2;
    ukf->Q[3*4+3] = sp.C3 * sp.C3;

    ukf->R[0] = sigma_camera * sigma_camera;

    /* ---- Initial state ---- */
    if (x0)
        memcpy(ukf->x, x0, UKF_N * sizeof(double));

    /* ---- Initial covariance ---- */
    if (P0_diag) {
        for (int i = 0; i < UKF_N; i++)
            ukf->P[i * UKF_N + i] = P0_diag[i];
    } else {
        ukf->P[0*4+0] = 10.0 * sp.C0 * sp.C0;
        ukf->P[1*4+1] = 10.0 * sp.C1 * sp.C1;
        ukf->P[2*4+2] = 10.0 * sp.C2 * sp.C2;
        ukf->P[3*4+3] = 10.0 * sp.C3 * sp.C3;
    }
}


/* ================================================================== */
/* ukf_predict — sigma-point time update                               */
/* ================================================================== */
/*
 * Step 1  Cholesky of (n+λ)·P  →  L   (using robust_chol)
 * Step 2  Generate 9 sigma points X[0..8] from x̂ and L
 * Step 3  Propagate each through f  →  X_pred[0..8]
 * Step 4  Recover prior mean:    x̂⁻ = Σ Wm[i] · X_pred[i]
 * Step 5  Recover prior cov:     P⁻  = Σ Wc[i] · d[i]·d[i]ᵀ + Q
 *         where  d[i] = X_pred[i] − x̂⁻
 *
 * X_pred is cached in the struct and reused by ukf_update.
 *
 * Returns 0 on success, -1 if P is not positive-definite.
 */
int ukf_predict(UKFState *ukf, const double *u, double dt)
{
    /* ---- Step 1: scale P and take Cholesky ---- */
    double scaled_P[UKF_N * UKF_N];
    double L       [UKF_N * UKF_N];

    double scale = (double)UKF_N + ukf->lambda;
    for (int i = 0; i < UKF_N * UKF_N; i++)
        scaled_P[i] = scale * ukf->P[i];

    /* robust_chol: tries without ε first, falls back to relative ε */
    if (robust_chol(scaled_P, L, UKF_N) != 0)
        return -1;

    /* ---- Step 2: build sigma-point matrix X [9×4]
     *
     * X[0]     = x̂
     * X[j+1]   = x̂ + L[:,j]       j = 0…3   (positive spread)
     * X[j+1+n] = x̂ − L[:,j]       j = 0…3   (negative spread)
     *
     * L is lower-triangular: L[r*N+j] is element (row r, col j).
     * Column j of L has entries only for rows r ≥ j.
     */
    double X[UKF_NSIGMA * UKF_N];
    memcpy(&X[0], ukf->x, UKF_N * sizeof(double));

    for (int j = 0; j < UKF_N; j++) {
        for (int r = 0; r < UKF_N; r++) {
            double col_j = L[r * UKF_N + j];
            X[(j + 1)        * UKF_N + r] = ukf->x[r] + col_j;
            X[(j + 1 + UKF_N) * UKF_N + r] = ukf->x[r] - col_j;
        }
    }

    /* ---- Step 3: propagate each sigma point through f ---- */
    for (int i = 0; i < UKF_NSIGMA; i++)
        f_lane(&X[i * UKF_N], u, dt, &ukf->X_pred[i * UKF_N]);

    /* ---- Step 4: predicted mean  x̂⁻ = Σ Wm[i] · X_pred[i] ---- */
    double x_prior[UKF_N];
    mat_zero(x_prior, UKF_N, 1);
    for (int i = 0; i < UKF_NSIGMA; i++)
        for (int j = 0; j < UKF_N; j++)
            x_prior[j] += ukf->Wm[i] * ukf->X_pred[i * UKF_N + j];

    /* ---- Step 5: predicted covariance
     *
     *   P⁻ = Σ Wc[i] · d[i]·d[i]ᵀ + Q      d[i] = X_pred[i] − x̂⁻
     *
     * Each term is an outer product accumulated with weight Wc[i].
     */
    double P_prior[UKF_N * UKF_N];
    mat_zero(P_prior, UKF_N, UKF_N);

    for (int i = 0; i < UKF_NSIGMA; i++) {
        double d[UKF_N];
        for (int j = 0; j < UKF_N; j++)
            d[j] = ukf->X_pred[i * UKF_N + j] - x_prior[j];

        for (int r = 0; r < UKF_N; r++)
            for (int c = 0; c < UKF_N; c++)
                P_prior[r * UKF_N + c] += ukf->Wc[i] * d[r] * d[c];
    }
    mat_add(P_prior, ukf->Q, P_prior, UKF_N, UKF_N);

    /* Commit predicted state — only reached on success */
    memcpy(ukf->x, x_prior,  UKF_N * sizeof(double));
    memcpy(ukf->P, P_prior,  UKF_N * UKF_N * sizeof(double));

    return 0;
}


/* ================================================================== */
/* ukf_update — sigma-point measurement update                         */
/* ================================================================== */
/*
 * Reuses ukf->X_pred from the immediately preceding ukf_predict call.
 *
 * Step 6  Z[i] = h(X_pred[i])                  — scalar, h(x)=C0=x[0]
 * Step 7  z̄   = Σ Wm[i] · Z[i]                 — predicted mean
 * Step 8  S    = Σ Wc[i] · (Z[i]−z̄)² + R      — innovation cov [1×1]
 * Step 9  Pxz  = Σ Wc[i] · (X_pred[i]−x̂⁻)·(Z[i]−z̄)ᵀ   — [4×1]
 * Step 10 K    = Pxz / S                         — [4×1]
 * Step 11 x̂   = x̂⁻ + K·(z − z̄)
 *         P    = P⁻ − K·S·Kᵀ
 *
 * Returns 0 on success, -1 if S is singular.
 */
int ukf_update(UKFState *ukf, const double *z)
{
    /* ---- Step 6: Z[i] = h(X_pred[i]) ---- */
    double Z[UKF_NSIGMA];
    for (int i = 0; i < UKF_NSIGMA; i++)
        Z[i] = h_lane(&ukf->X_pred[i * UKF_N]);

    /* ---- Step 7: z̄ (scalar) ---- */
    double z_bar = 0.0;
    for (int i = 0; i < UKF_NSIGMA; i++)
        z_bar += ukf->Wm[i] * Z[i];

    /* ---- Step 8: S = Σ Wc[i]·(Z[i]−z̄)² + R ---- */
    double S_val = ukf->R[0];
    for (int i = 0; i < UKF_NSIGMA; i++) {
        double dz = Z[i] - z_bar;
        S_val += ukf->Wc[i] * dz * dz;
    }
    ukf->S[0] = S_val;

    if (fabs(S_val) < 1e-12)
        return -1;

    double S_inv = 1.0 / S_val;

    /* ---- NIS = (z − z̄)² / S ---- */
    double innov = z[0] - z_bar;
    ukf->innovation[0] = innov;
    ukf->NIS           = innov * innov * S_inv;

    /* ---- Step 9: Pxz = Σ Wc[i]·(X_pred[i]−x̂⁻)·(Z[i]−z̄)ᵀ  [4×1]
     *
     * For m=1: (Z[i]−z̄)ᵀ is a scalar, so each term is Wc[i]·dz[i]·dx[i].
     */
    double Pxz[UKF_N];
    mat_zero(Pxz, UKF_N, 1);

    for (int i = 0; i < UKF_NSIGMA; i++) {
        double dz = Z[i] - z_bar;
        for (int j = 0; j < UKF_N; j++) {
            double dx = ukf->X_pred[i * UKF_N + j] - ukf->x[j];
            Pxz[j] += ukf->Wc[i] * dx * dz;
        }
    }

    /* ---- Step 10: K = Pxz · S⁻¹  [4×1] = [4×1] × scalar ---- */
    for (int j = 0; j < UKF_N; j++)
        ukf->K[j] = Pxz[j] * S_inv;

    /* ---- Step 11a: x̂ = x̂⁻ + K·(z − z̄) ---- */
    for (int j = 0; j < UKF_N; j++)
        ukf->x[j] += ukf->K[j] * innov;

    /* ---- Step 11b: P = P⁻ − K·S·Kᵀ
     *
     * For m=1: K·S·Kᵀ = S_val · outer(K, K)
     */
    for (int r = 0; r < UKF_N; r++)
        for (int c = 0; c < UKF_N; c++)
            ukf->P[r * UKF_N + c] -= S_val * ukf->K[r] * ukf->K[c];

    mat_symmetrise(ukf->P, UKF_N);

    return 0;
}


/* ================================================================== */
/* ukf_print                                                           */
/* ================================================================== */

void ukf_print(const UKFState *ukf)
{
    printf("UKF State:\n");
    printf("  x\u0302  = [C0=%8.4f m  C1=%8.5f rad  C2=%9.6f 1/m  C3=%10.7f 1/m\u00b2]\n",
           ukf->x[0], ukf->x[1], ukf->x[2], ukf->x[3]);
    printf("  P diag = [%8.5f  %8.5f  %8.5f  %8.5f]\n",
           ukf->P[0*4+0], ukf->P[1*4+1], ukf->P[2*4+2], ukf->P[3*4+3]);
    printf("  NIS    = %.4f   innovation = %.4f m\n",
           ukf->NIS, ukf->innovation[0]);
    printf("  \u03bb = %.6f   Wm[0]=%.4f  Wm[1..8]=%.4f\n",
           ukf->lambda, ukf->Wm[0], ukf->Wm[1]);
}
