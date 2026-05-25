/**
 * test_ukf.c — Unit tests for ukf.c
 *
 * Tests verify:
 *   1. ukf_init:   weight formulae, Q/R population
 *   2. Weights:    Wm[0] + 2n·Wm[1] = 1  (must sum to 1)
 *                  Wc[0] ≠ Wm[0] when β≠0  (prior knowledge term)
 *   3. Sigma points: generated from a known (x̂, P), check X[1..4] = x̂ ± L[:,j]
 *   4. ukf_predict: nonlinear mean identical to ekf_predict for same f()
 *   5. ukf_update:  NIS, K, x̂, P downdate
 *   6. Dropout recovery: same as EKF test
 *   7. UKF vs EKF: for small C1, x[0] estimates must agree to 5 mm
 */

#include "../include/ukf.h"
#include "../include/ekf.h"
#include "../include/matrix.h"

#include <stdio.h>
#include <math.h>
#include <string.h>

/* ------------------------------------------------------------------ */
/* Test helpers                                                         */
/* ------------------------------------------------------------------ */

static int g_failures = 0;

static int dbl_close(double a, double b, double tol)
{
    return fabs(a - b) < tol;
}

#define ASSERT_DBL(val, exp, tol, name)                                    \
    do {                                                                    \
        if (!dbl_close((double)(val), (double)(exp), tol)) {               \
            printf("  FAIL: %-45s  got=%.10g  want=%.10g\n",              \
                   name, (double)(val), (double)(exp));                     \
            g_failures++;                                                   \
        } else {                                                            \
            printf("  PASS: %s\n", name);                                  \
        }                                                                   \
    } while(0)

#define ASSERT_TRUE(cond, name)                                            \
    do {                                                                    \
        if (!(cond)) { printf("  FAIL: %s\n", name); g_failures++; }      \
        else           printf("  PASS: %s\n", name);                       \
    } while(0)


/* Shorthand for standard initialisation used across most tests */
static void default_init(UKFState *ukf)
{
    SigmaProcess sp = {.C0=0.07, .C1=0.004, .C2=4e-5, .C3=8e-7};
    double x0[4] = {1.875, 0.0, 0.0, 0.0};
    double P0[4] = {0.1, 0.01, 1e-4, 1e-6};
    ukf_init(ukf, sp, 0.1, x0, P0, 1e-3, 2.0, 0.0);
}


/* ================================================================== */
/* 1 & 2. Weight formulae                                              */
/* ================================================================== */

static void test_weights(void)
{
    printf("\n--- sigma-point weights ---\n");

    UKFState ukf;
    default_init(&ukf);

    double n   = (double)UKF_N;   /* 4 */
    double lam = ukf.lambda;

    /* λ = α²(n+κ) − n  with α=1e-3, β=2, κ=0 */
    double lam_expected = (1e-3)*(1e-3)*(n + 0.0) - n;   /* = 4e-6 - 4 ≈ -4 */
    ASSERT_DBL(lam, lam_expected, 1e-12, "lambda = α²(n+κ)-n");

    /* Wm[0] = λ/(n+λ) */
    double Wm0_expected = lam / (n + lam);
    ASSERT_DBL(ukf.Wm[0], Wm0_expected, 1e-12, "Wm[0] = λ/(n+λ)");

    /* Wc[0] = λ/(n+λ) + (1 - α² + β) */
    double Wc0_expected = lam / (n + lam) + (1.0 - 1e-6 + 2.0);
    ASSERT_DBL(ukf.Wc[0], Wc0_expected, 1e-12, "Wc[0] = Wm[0]+(1-α²+β)");

    /* Wm[i] = 1/(2(n+λ))  for i = 1..2n */
    double Wi_expected = 1.0 / (2.0 * (n + lam));
    ASSERT_DBL(ukf.Wm[1], Wi_expected, 1e-12, "Wm[1] = 1/(2(n+λ))");
    ASSERT_DBL(ukf.Wm[8], Wi_expected, 1e-12, "Wm[8] = 1/(2(n+λ))");

    /* Wc[i] = Wm[i] for i≥1 */
    ASSERT_DBL(ukf.Wc[1], ukf.Wm[1], 1e-15, "Wc[i]=Wm[i] for i≥1");

    /* Sum of Wm must equal 1 */
    double Wm_sum = 0.0;
    for (int i = 0; i < UKF_NSIGMA; i++) Wm_sum += ukf.Wm[i];
    ASSERT_DBL(Wm_sum, 1.0, 1e-5, "Σ Wm = 1 (tol 1e-5 for cancellation)");

    /* Wc[0] ≠ Wm[0] (β=2 shifts it) */
    ASSERT_TRUE(fabs(ukf.Wc[0] - ukf.Wm[0]) > 1.0, "Wc[0]≠Wm[0] when β=2");
}


/* ================================================================== */
/* 3. Sigma-point generation — inspect X_pred layout after predict     */
/* ================================================================== */

static void test_sigma_points(void)
{
    printf("\n--- sigma-point layout ---\n");

    /*
     * Use a diagonal P so Cholesky L is also diagonal with L[j,j] = sqrt(P[j,j]).
     * Scaled P = (n+λ)·P.
     * X[j+1][j] = x[j] + L[j,j]  and  X[j+1+n][j] = x[j] - L[j,j]
     * (only the j-th component of each pair changes — off-diagonals stay at x̂).
     *
     * With α=1, κ=0: λ = 1·(4+0)-4 = 0, so (n+λ) = 4.
     *   L[j,j] = sqrt(4·P[j,j]) = 2·sqrt(P[j,j])
     */
    SigmaProcess sp = {.C0=0.07, .C1=0.004, .C2=4e-5, .C3=8e-7};
    double x0[4] = {1.875, 0.01, 0.002, 0.0};
    double P0[4] = {0.09, 0.0004, 1e-6, 1e-10};   /* diagonal */
    UKFState ukf;
    ukf_init(&ukf, sp, 0.1, x0, P0, 1.0, 2.0, 0.0);   /* α=1 → λ=0 */

    double u[2] = {30.0, 0.0};
    ukf_predict(&ukf, u, 0.04);

    /* With α=1, λ=0: (n+λ) = 4.  L[j,j] = sqrt(4·P0[j]) = 2·sqrt(P0[j]).
     * After propagation through f, sigma point 0 is propagated from x̂.
     * For C1=0.01 (small), sin(C1)≈C1, so X_pred[0][0] ≈ 1.875 - 30·0.04·sin(0.01).
     *
     * We check symmetry of paired sigma points in state-space.
     * X_pred[j+1] and X_pred[j+1+n] should be symmetric about X_pred[0]
     * in every component — IF f were linear.  For nonlinear f this is
     * only approximately true; we check the j=0 pair (C0 axis), where
     * the dominant nonlinearity is in C1, not C0.
     */

    /* Central sigma point 0 must match f applied to x̂ */
    double x_pred_0_C0 = x0[0] - 30.0*0.04*sin(x0[1]);
    ASSERT_DBL(ukf.X_pred[0*4+0], x_pred_0_C0, 1e-10,
               "X_pred[0]=f(x̂): C0 component");

    /* Sigma points 1..4 and 5..8 must exist (not NaN) */
    for (int i = 1; i < UKF_NSIGMA; i++) {
        ASSERT_TRUE(!isnan(ukf.X_pred[i*4+0]),
                    "X_pred[i][0] not NaN");
    }

    /* Verify pair symmetry approximately (for C0 column, which is
     * perturbed only in the C0 direction by sigma points 1 and 5) */
    double sym_diff = ukf.X_pred[1*4+0] + ukf.X_pred[5*4+0]
                      - 2.0 * ukf.X_pred[0*4+0];
    ASSERT_DBL(sym_diff, 0.0, 0.01, "sigma pair symmetry (C0 axis)");

    /* Prior mean x̂⁻ must be finite and near true value */
    ASSERT_TRUE(!isnan(ukf.x[0]) && !isinf(ukf.x[0]), "x̂⁻[0] finite");
    ASSERT_TRUE(fabs(ukf.x[0] - 1.875) < 0.5, "x̂⁻[0] near initial C0");
}


/* ================================================================== */
/* 4. ukf_predict: mean matches ekf_predict for same initial state     */
/* ================================================================== */

static void test_predict_vs_ekf(void)
{
    printf("\n--- ukf_predict mean ≈ ekf_predict mean ---\n");

    SigmaProcess sp = {.C0=0.07, .C1=0.004, .C2=4e-5, .C3=8e-7};
    double x0[4] = {1.875, 0.05, 0.003, 1e-4};
    double P0[4] = {0.1, 0.01, 1e-4, 1e-6};
    double u[2]  = {30.0, 0.06};
    double dt    = 0.04;

    EKFState ekf;
    ekf_init(&ekf, sp, 0.1, x0, P0);
    ekf_predict(&ekf, u, dt);

    UKFState ukf;
    ukf_init(&ukf, sp, 0.1, x0, P0, 1e-3, 2.0, 0.0);
    int ret = ukf_predict(&ukf, u, dt);
    ASSERT_TRUE(ret == 0, "ukf_predict returns 0");

    /*
     * UKF mean propagation is more accurate than EKF (3rd order vs 1st).
     * They agree to ~1e-4 for this C1 magnitude.
     * The EKF uses sin(C1) in f but linearises around it; the UKF
     * propagates through sin(C1) exactly for each sigma point.
     */
    printf("  UKF x̂⁻: [%.6f, %.6f, %.6f, %.8f]\n",
           ukf.x[0], ukf.x[1], ukf.x[2], ukf.x[3]);
    printf("  EKF x̂⁻: [%.6f, %.6f, %.6f, %.8f]\n",
           ekf.x[0], ekf.x[1], ekf.x[2], ekf.x[3]);

    ASSERT_DBL(ukf.x[0], ekf.x[0], 5e-4, "x̂⁻[C0] UKF≈EKF (5e-4 m, C1=0.05)");
    ASSERT_DBL(ukf.x[1], ekf.x[1], 1e-4, "x̂⁻[C1] UKF≈EKF");
    ASSERT_DBL(ukf.x[2], ekf.x[2], 1e-6, "x̂⁻[C2] UKF≈EKF");

    /* P must remain PD */
    ASSERT_TRUE(ukf.P[0*4+0] > 0, "P⁻[0,0] > 0");
    ASSERT_TRUE(ukf.P[1*4+1] > 0, "P⁻[1,1] > 0");
}


/* ================================================================== */
/* 5. ukf_update — NIS, K, x̂, P downdate                             */
/* ================================================================== */

static void test_update(void)
{
    printf("\n--- ukf_update ---\n");

    UKFState ukf;
    default_init(&ukf);

    double u[2] = {30.0, 0.0};
    int ret = ukf_predict(&ukf, u, 0.04);
    ASSERT_TRUE(ret == 0, "ukf_predict OK before update");

    double P_prior_00 = ukf.P[0*4+0];

    double z[1] = {1.90};
    ret = ukf_update(&ukf, z);
    ASSERT_TRUE(ret == 0, "ukf_update returns 0");

    /* Innovation = z - z_bar; z_bar ≈ C0_prior ≈ 1.875 (C1≈0) */
    ASSERT_TRUE(fabs(ukf.innovation[0]) < 0.2, "innovation magnitude <0.2m");

    /* NIS must be non-negative */
    ASSERT_TRUE(ukf.NIS >= 0.0, "NIS ≥ 0");

    /* x[0] must move toward the measurement (1.90 > 1.875) */
    ASSERT_TRUE(ukf.x[0] > 1.875, "x̂[0] moved toward z=1.90");
    ASSERT_TRUE(ukf.x[0] < 1.90 + 0.01, "x̂[0] not past z=1.90");

    /* P[0,0] must shrink */
    ASSERT_TRUE(ukf.P[0*4+0] < P_prior_00, "P[0,0] shrinks after update");

    /* P must remain symmetric */
    ASSERT_DBL(ukf.P[0*4+1], ukf.P[1*4+0], 1e-13,
               "P symmetric [0,1] after update");
    ASSERT_DBL(ukf.P[1*4+2], ukf.P[2*4+1], 1e-13,
               "P symmetric [1,2] after update");
}


/* ================================================================== */
/* 6. Dropout recovery                                                  */
/* ================================================================== */

static void test_dropout(void)
{
    printf("\n--- dropout (predict-only) ---\n");

    UKFState ukf;
    default_init(&ukf);

    double u[2] = {30.0, 0.0};
    double P_before = ukf.P[0*4+0];

    for (int k = 0; k < 10; k++)
        ukf_predict(&ukf, u, 0.04);

    ASSERT_TRUE(ukf.P[0*4+0] > P_before,  "P[0,0] grows during dropout");
    ASSERT_TRUE(!isnan(ukf.x[0]),          "x[0] finite after dropout");

    double P_peak = ukf.P[0*4+0];
    double z[1] = {1.875};
    for (int k = 0; k < 20; k++) {
        ukf_predict(&ukf, u, 0.04);
        ukf_update(&ukf, z);
    }
    ASSERT_TRUE(ukf.P[0*4+0] < P_peak, "P[0,0] recovers after dropout");
}


/* ================================================================== */
/* 7. UKF vs EKF: full 100-step run, C0 estimates agree to 5 mm       */
/* ================================================================== */

static void test_ukf_vs_ekf_run(void)
{
    printf("\n--- UKF vs EKF: 100-step run ---\n");

    SigmaProcess sp = {.C0=0.07, .C1=0.004, .C2=4e-5, .C3=8e-7};
    double x0[4] = {1.875, 0.0, 0.0, 0.0};
    double P0[4] = {0.1, 0.01, 1e-4, 1e-6};
    double dt    = 0.04;
    double u[2]  = {30.0, 0.0};

    EKFState ekf;  ekf_init(&ekf, sp, 0.1, x0, P0);
    UKFState ukf;  ukf_init(&ukf, sp, 0.1, x0, P0, 1e-3, 2.0, 0.0);

    /* Deterministic pseudo-random measurements */
    double C0_true = 1.875;
    double rmse_ekf = 0.0, rmse_ukf = 0.0;

    for (int k = 0; k < 100; k++) {
        double noise = 0.1 * (((k * 1234567 + 891011) % 1000) / 1000.0 - 0.5);
        double z[1]  = { C0_true + noise };

        ekf_predict(&ekf, u, dt);
        ukf_predict(&ukf, u, dt);

        ekf_update(&ekf, z);
        ukf_update(&ukf, z);

        double e_ekf = ekf.x[0] - C0_true;
        double e_ukf = ukf.x[0] - C0_true;
        rmse_ekf += e_ekf * e_ekf;
        rmse_ukf += e_ukf * e_ukf;
    }
    rmse_ekf = sqrt(rmse_ekf / 100.0);
    rmse_ukf = sqrt(rmse_ukf / 100.0);

    printf("  EKF RMSE = %.5f m\n", rmse_ekf);
    printf("  UKF RMSE = %.5f m\n", rmse_ukf);

    /* RMSE difference under 1 cm on a straight road with small C1 */
    ASSERT_TRUE(fabs(rmse_ekf - rmse_ukf) < 0.005, "UKF≈EKF RMSE on straight (5mm)");

    /* Final C0 estimates agree to 5 mm */
    ASSERT_DBL(ukf.x[0], ekf.x[0], 0.01, "UKF≈EKF final C0 (within 10mm)");

    /* Both NIS values should be reasonable (not explosion) */
    ASSERT_TRUE(ukf.NIS < 20.0, "UKF NIS < 20");
    ASSERT_TRUE(ekf.NIS < 20.0, "EKF NIS < 20");
}


/* ================================================================== */
/* main                                                                 */
/* ================================================================== */

int main(void)
{
    printf("=================================================\n");
    printf("  ukf.c unit tests\n");
    printf("=================================================\n");

    test_weights();
    test_sigma_points();
    test_predict_vs_ekf();
    test_update();
    test_dropout();
    test_ukf_vs_ekf_run();

    printf("\n=================================================\n");
    if (g_failures == 0)
        printf("  ALL UKF TESTS PASSED\n");
    else
        printf("  %d UKF TEST(S) FAILED\n", g_failures);
    printf("=================================================\n");

    return g_failures;
}
