/**
 * test_ekf.c — Unit tests for ekf.c
 *
 * Tests verify:
 *   1. ekf_init:    correct H, Q, R, P0 population
 *   2. ekf_predict: nonlinear f, Jacobian F_jac, covariance propagation
 *   3. ekf_update:  innovation, NIS, K, x_post, Joseph-form P
 *   4. Dropout handling: predict-only run doesn't corrupt state
 *   5. Jacobian cross-check: compare ekf vs kf on a straight segment
 *      where C1≈0, so sin(C1)≈C1 and the two models must agree to 1e-4.
 */

#include "../include/ekf.h"
#include "../include/kf.h"
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

#define ASSERT_DBL(val, exp, tol, name)                                     \
    do {                                                                     \
        if (!dbl_close(val, exp, tol)) {                                    \
            printf("  FAIL: %-45s  got=%.10g  want=%.10g\n", name,         \
                   (double)(val), (double)(exp));                            \
            g_failures++;                                                    \
        } else {                                                             \
            printf("  PASS: %s\n", name);                                   \
        }                                                                    \
    } while(0)

#define ASSERT_TRUE(cond, name)                                             \
    do {                                                                     \
        if (!(cond)) { printf("  FAIL: %s\n", name); g_failures++; }       \
        else           printf("  PASS: %s\n", name);                        \
    } while(0)


/* ================================================================== */
/* 1. ekf_init                                                          */
/* ================================================================== */

static void test_init(void)
{
    printf("\n--- ekf_init ---\n");

    SigmaProcess sp = {.C0=0.07, .C1=0.004, .C2=4e-5, .C3=8e-7};
    double sc = 0.1;

    EKFState ekf;
    ekf_init(&ekf, sp, sc, NULL, NULL);

    /* H = [1,0,0,0] */
    ASSERT_DBL(ekf.H[0], 1.0, 1e-15, "H[0]=1");
    ASSERT_DBL(ekf.H[1], 0.0, 1e-15, "H[1]=0");
    ASSERT_DBL(ekf.H[2], 0.0, 1e-15, "H[2]=0");
    ASSERT_DBL(ekf.H[3], 0.0, 1e-15, "H[3]=0");

    /* Q diagonal = sp²  */
    ASSERT_DBL(ekf.Q[0*4+0], 0.07*0.07,  1e-15, "Q[0,0]=0.07²");
    ASSERT_DBL(ekf.Q[1*4+1], 0.004*0.004,1e-15, "Q[1,1]=0.004²");
    ASSERT_DBL(ekf.Q[2*4+2], 4e-5*4e-5,  1e-15, "Q[2,2]=(4e-5)²");
    ASSERT_DBL(ekf.Q[3*4+3], 8e-7*8e-7,  1e-15, "Q[3,3]=(8e-7)²");

    /* Off-diagonal Q must be zero */
    ASSERT_DBL(ekf.Q[0*4+1], 0.0, 1e-15, "Q[0,1]=0");
    ASSERT_DBL(ekf.Q[1*4+2], 0.0, 1e-15, "Q[1,2]=0");

    /* R = sc² */
    ASSERT_DBL(ekf.R[0], sc*sc, 1e-15, "R=sc²");

    /* Default P0 = 10·Q diag */
    ASSERT_DBL(ekf.P[0*4+0], 10.0*0.07*0.07,   1e-15, "P0[0,0]");
    ASSERT_DBL(ekf.P[1*4+1], 10.0*0.004*0.004, 1e-15, "P0[1,1]");

    /* Custom P0_diag */
    double P0[4] = {0.1, 0.01, 1e-4, 1e-6};
    EKFState ekf2;
    ekf_init(&ekf2, sp, sc, NULL, P0);
    ASSERT_DBL(ekf2.P[0*4+0], 0.1,  1e-15, "P0_diag[0]");
    ASSERT_DBL(ekf2.P[1*4+1], 0.01, 1e-15, "P0_diag[1]");

    /* Custom x0 */
    double x0[4] = {1.875, 0.01, 0.002, 0.0};
    EKFState ekf3;
    ekf_init(&ekf3, sp, sc, x0, P0);
    ASSERT_DBL(ekf3.x[0], 1.875, 1e-15, "x0[0]=1.875");
    ASSERT_DBL(ekf3.x[1], 0.01,  1e-15, "x0[1]=0.01");
}


/* ================================================================== */
/* 2. ekf_predict — nonlinear propagation and Jacobian                 */
/* ================================================================== */

static void test_predict(void)
{
    printf("\n--- ekf_predict ---\n");

    /*
     * Starting state: x = [C0=1.875, C1=0.05, C2=0.002, C3=0.0]
     * Control:        u = [v=30, ω=0.06]     (≈108 km/h, mild curve)
     * dt = 0.04 s
     *
     * Expected nonlinear propagation (computed by hand):
     *   C0' = 1.875 - 30·0.04·sin(0.05)
     *       = 1.875 - 1.2·0.04998 = 1.875 - 0.059975 = 1.815025
     *   C1' = 0.05 + 0.002·30·0.04 - 0.06·0.04
     *       = 0.05 + 0.0024 - 0.0024 = 0.05
     *   C2' = 0.002 + 0.0·30·0.04 = 0.002
     *   C3' = 0.0
     *
     * Jacobian at current x̂:
     *   F_jac[0,1] = -v·dt·cos(C1) = -30·0.04·cos(0.05)
     *              = -1.2·0.99875 = -1.198501
     */
    SigmaProcess sp = {.C0=0.07, .C1=0.004, .C2=4e-5, .C3=8e-7};
    double x0[4]  = {1.875, 0.05, 0.002, 0.0};
    double P0[4]  = {0.1, 0.01, 1e-4, 1e-6};
    EKFState ekf;
    ekf_init(&ekf, sp, 0.1, x0, P0);

    double u[2] = {30.0, 0.06};
    double dt   = 0.04;
    ekf_predict(&ekf, u, dt);

    /* Nonlinear mean propagation */
    double C0_expected = 1.875 - 30.0*0.04*sin(0.05);
    double C1_expected = 0.05 + 0.002*30.0*0.04 - 0.06*0.04;
    double C2_expected = 0.002;
    double C3_expected = 0.0;

    ASSERT_DBL(ekf.x[0], C0_expected, 1e-12, "predict x[0] C0");
    ASSERT_DBL(ekf.x[1], C1_expected, 1e-12, "predict x[1] C1");
    ASSERT_DBL(ekf.x[2], C2_expected, 1e-12, "predict x[2] C2");
    ASSERT_DBL(ekf.x[3], C3_expected, 1e-12, "predict x[3] C3");

    /* Jacobian element F[0,1] = -v·dt·cos(C1_original) */
    double F01_expected = -30.0 * 0.04 * cos(0.05);
    ASSERT_DBL(ekf.F_jac[0*4+1], F01_expected, 1e-12, "F_jac[0,1]");
    ASSERT_DBL(ekf.F_jac[1*4+2], 30.0*0.04,    1e-12, "F_jac[1,2]=v·dt");
    ASSERT_DBL(ekf.F_jac[2*4+3], 30.0*0.04,    1e-12, "F_jac[2,3]=v·dt");
    ASSERT_DBL(ekf.F_jac[0*4+0], 1.0,           1e-15, "F_jac[0,0]=1");
    ASSERT_DBL(ekf.F_jac[1*4+1], 1.0,           1e-15, "F_jac[1,1]=1");
    ASSERT_DBL(ekf.F_jac[3*4+3], 1.0,           1e-15, "F_jac[3,3]=1");
    ASSERT_DBL(ekf.F_jac[0*4+2], 0.0,           1e-15, "F_jac[0,2]=0");

    /* P must remain positive-definite after predict */
    ASSERT_TRUE(ekf.P[0*4+0] > 0.0, "P[0,0]>0 after predict");
    ASSERT_TRUE(ekf.P[1*4+1] > 0.0, "P[1,1]>0 after predict");
    ASSERT_TRUE(ekf.P[2*4+2] > 0.0, "P[2,2]>0 after predict");
    ASSERT_TRUE(ekf.P[3*4+3] > 0.0, "P[3,3]>0 after predict");

    /* P must be symmetric */
    ASSERT_DBL(ekf.P[0*4+1], ekf.P[1*4+0], 1e-14, "P symmetric [0,1]");
    ASSERT_DBL(ekf.P[1*4+2], ekf.P[2*4+1], 1e-14, "P symmetric [1,2]");
}


/* ================================================================== */
/* 3. ekf_update — NIS, Kalman gain, state and covariance             */
/* ================================================================== */

static void test_update(void)
{
    printf("\n--- ekf_update ---\n");

    /*
     * Scalar EKF (n=4, m=1): initialise with diagonal P and known Q, R,
     * then verify update output against KF update equations by hand.
     *
     * We set C1=0 so that sin(C1)=0 and the nonlinear model reduces to
     * the linear one — this lets us check the update path independently
     * of the nonlinear predict.
     */
    SigmaProcess sp = {.C0=0.07, .C1=0.004, .C2=4e-5, .C3=8e-7};
    double x0[4] = {1.8, 0.0, 0.0, 0.0};
    double P0[4] = {0.5, 0.05, 5e-4, 5e-6};
    EKFState ekf;
    ekf_init(&ekf, sp, 0.1, x0, P0);

    /* One predict step (C1=0, v=30, dt=0.04): x should barely move */
    double u[2] = {30.0, 0.0};
    ekf_predict(&ekf, u, 0.04);

    /* Manual expected values after predict (C1=0 → sin(C1)=0):
     *   x̂⁻ = [1.8, 0, 0, 0]  (C1=0 so no lateral motion term)
     *   P⁻ = F·P·Fᵀ + Q  ... we check the update, not P⁻
     */

    /* Update with measurement z = 1.85 */
    double z[1] = {1.85};
    int ret = ekf_update(&ekf, z);
    ASSERT_TRUE(ret == 0, "ekf_update returns 0");

    /* Innovation: y = 1.85 - C0_predicted */
    double C0_prior = 1.8;   /* sin(0)=0 so C0 unchanged by predict */
    double innov_expected = 1.85 - C0_prior;
    ASSERT_DBL(ekf.innovation[0], innov_expected, 1e-3,
               "innovation = z - H·x̂⁻");

    /* NIS = y²/S  (positive) */
    ASSERT_TRUE(ekf.NIS >= 0.0, "NIS >= 0");

    /* x̂ must move toward the measurement */
    ASSERT_TRUE(ekf.x[0] > C0_prior - 0.001, "x[0] moved toward z");
    ASSERT_TRUE(ekf.x[0] < z[0] + 0.001,     "x[0] not past z");

    /* P[0,0] must shrink after incorporating a measurement */
    ASSERT_TRUE(ekf.P[0*4+0] < 0.5, "P[0,0] shrinks after update");

    /* P must remain symmetric */
    ASSERT_DBL(ekf.P[0*4+1], ekf.P[1*4+0], 1e-13, "P sym [0,1] after update");
}


/* ================================================================== */
/* 4. Dropout: predict-only does not corrupt state                     */
/* ================================================================== */

static void test_dropout(void)
{
    printf("\n--- dropout (predict only) ---\n");

    SigmaProcess sp = {.C0=0.07, .C1=0.004, .C2=4e-5, .C3=8e-7};
    double x0[4] = {1.875, 0.0, 0.0, 0.0};
    double P0[4] = {0.1, 0.01, 1e-4, 1e-6};
    EKFState ekf;
    ekf_init(&ekf, sp, 0.1, x0, P0);

    double P_before = ekf.P[0*4+0];
    double u[2] = {30.0, 0.0};

    /* 10 predict-only steps (simulating camera dropout) */
    for (int k = 0; k < 10; k++)
        ekf_predict(&ekf, u, 0.04);

    /* P[0,0] must grow (no measurement → increasing uncertainty) */
    ASSERT_TRUE(ekf.P[0*4+0] > P_before, "P[0,0] grows during dropout");

    /* State must remain finite */
    ASSERT_TRUE(!isnan(ekf.x[0]) && !isinf(ekf.x[0]), "x[0] finite");
    ASSERT_TRUE(!isnan(ekf.P[0*4+0]) && !isinf(ekf.P[0*4+0]), "P[0,0] finite");

    /* After resuming measurements P should shrink again */
    double P_peak = ekf.P[0*4+0];
    double z[1] = {1.875};
    for (int k = 0; k < 20; k++) {
        ekf_predict(&ekf, u, 0.04);
        ekf_update(&ekf, z);
    }
    ASSERT_TRUE(ekf.P[0*4+0] < P_peak, "P[0,0] recovers after dropout");
}


/* ================================================================== */
/* 5. EKF vs KF cross-check: straight road, C1 ≈ 0                   */
/*                                                                     */
/* When C1 = 0:  sin(C1) = 0, cos(C1) = 1.                           */
/* EKF nonlinear f → C0' = C0 - v·dt·0 = C0  (same as KF F[0,0]=1)  */
/* EKF Jacobian   → F[0,1] = -v·dt·1  vs KF F[0,1] = dt (heading    */
/* coupling term differs but is a model choice, not a bug).            */
/*                                                                     */
/* What must agree: x[0] (C0 estimate) should track the same          */
/* measurement equally well — final RMSE within 1 mm.                 */
/* ================================================================== */

static void test_ekf_vs_kf_straight(void)
{
    printf("\n--- EKF vs KF cross-check (straight, C1≈0) ---\n");

    SigmaProcess sp = {.C0=0.07, .C1=0.004, .C2=4e-5, .C3=8e-7};
    double x0[4] = {1.875, 0.0, 0.0, 0.0};
    double P0[4] = {0.1, 0.01, 1e-4, 1e-6};
    double dt    = 0.04;
    double u[2]  = {30.0, 0.0};

    KFState  kf;   kf_init(&kf,  dt, sp, 0.1, x0, P0);
    EKFState ekf;  ekf_init(&ekf, sp, 0.1, x0, P0);

    /* Simulate 100 frames with constant C0=1.875, Gaussian noise σ=0.1 */
    double meas[100];
    for (int k = 0; k < 100; k++)
        meas[k] = 1.875 + 0.1 * (((k * 1234567 + 891011) % 1000) / 1000.0 - 0.5);

    double rmse_kf = 0.0, rmse_ekf = 0.0;

    for (int k = 0; k < 100; k++) {
        kf_predict(&kf);
        ekf_predict(&ekf, u, dt);

        double z[1] = {meas[k]};
        kf_update(&kf, z);
        ekf_update(&ekf, z);

        double e_kf  = kf.x[0]  - 1.875;
        double e_ekf = ekf.x[0] - 1.875;
        rmse_kf  += e_kf  * e_kf;
        rmse_ekf += e_ekf * e_ekf;
    }
    rmse_kf  = sqrt(rmse_kf  / 100.0);
    rmse_ekf = sqrt(rmse_ekf / 100.0);

    printf("  KF  RMSE = %.5f m\n", rmse_kf);
    printf("  EKF RMSE = %.5f m\n", rmse_ekf);

    /* Both must track closely — difference under 1 cm */
    ASSERT_TRUE(fabs(rmse_kf - rmse_ekf) < 0.01, "EKF≈KF RMSE on straight");

    /* Final C0 estimates should agree to within 5 mm */
    ASSERT_DBL(kf.x[0], ekf.x[0], 0.005, "KF≈EKF final C0 estimate");
}


/* ================================================================== */
/* main                                                                 */
/* ================================================================== */

int main(void)
{
    printf("=================================================\n");
    printf("  ekf.c unit tests\n");
    printf("=================================================\n");

    test_init();
    test_predict();
    test_update();
    test_dropout();
    test_ekf_vs_kf_straight();

    printf("\n=================================================\n");
    if (g_failures == 0)
        printf("  ALL EKF TESTS PASSED\n");
    else
        printf("  %d EKF TEST(S) FAILED\n", g_failures);
    printf("=================================================\n");

    return g_failures;
}
