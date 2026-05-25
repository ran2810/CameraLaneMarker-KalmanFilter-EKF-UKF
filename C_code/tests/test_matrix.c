/**
 * test_matrix.c — Unit tests for matrix.c
 *
 * Each test is self-contained, prints PASS/FAIL, and returns a count
 * of failures.  main() returns the total failure count as the exit code
 * (so `make test` can detect failures via $?).
 *
 * Tests use known-answer results computed by hand or verified in Python.
 */

#include "../include/matrix.h"

#include <stdio.h>
#include <math.h>
#include <string.h>

/* ------------------------------------------------------------------ */
/* Test helpers                                                         */
/* ------------------------------------------------------------------ */

#define MAX_FAILURES  100
static int g_failures = 0;

static int nearly_equal(double a, double b, double tol)
{
    return fabs(a - b) < tol;
}

/* Check all elements of two flat arrays of length n. */
static int arrays_close(const double *A, const double *B, size_t n, double tol)
{
    for (size_t i = 0; i < n; i++)
        if (!nearly_equal(A[i], B[i], tol)) return 0;
    return 1;
}

#define ASSERT_CLOSE(A, B, n, tol, name)  \
    do {                                   \
        if (!arrays_close(A, B, n, tol)) { \
            printf("  FAIL: %s\n", name);  \
            g_failures++;                  \
        } else {                           \
            printf("  PASS: %s\n", name);  \
        }                                  \
    } while(0)

#define ASSERT_INT_EQ(val, expected, name)            \
    do {                                               \
        if ((val) != (expected)) {                     \
            printf("  FAIL: %s  (got %d)\n", name, val); \
            g_failures++;                              \
        } else {                                       \
            printf("  PASS: %s\n", name);             \
        }                                              \
    } while(0)

#define ASSERT_SCALAR(val, expected, tol, name)       \
    do {                                               \
        if (!nearly_equal(val, expected, tol)) {       \
            printf("  FAIL: %s  (got %.8g, want %.8g)\n", name, (double)(val), (double)(expected)); \
            g_failures++;                              \
        } else {                                       \
            printf("  PASS: %s\n", name);             \
        }                                              \
    } while(0)


/* ================================================================== */
/* Test: mat_zero / mat_eye                                            */
/* ================================================================== */

static void test_zero_eye(void)
{
    printf("\n--- mat_zero / mat_eye ---\n");

    double A[9];
    mat_zero(A, 3, 3);
    double zeros[9] = {0};
    ASSERT_CLOSE(A, zeros, 9, 1e-15, "mat_zero 3×3");

    double I[9];
    mat_eye(I, 3);
    double I_expected[9] = {1,0,0, 0,1,0, 0,0,1};
    ASSERT_CLOSE(I, I_expected, 9, 1e-15, "mat_eye 3×3");
}


/* ================================================================== */
/* Test: mat_add / mat_sub                                             */
/* ================================================================== */

static void test_add_sub(void)
{
    printf("\n--- mat_add / mat_sub ---\n");

    double A[4] = {1, 2, 3, 4};
    double B[4] = {5, 6, 7, 8};
    double C[4];

    mat_add(A, B, C, 2, 2);
    double expected_add[4] = {6, 8, 10, 12};
    ASSERT_CLOSE(C, expected_add, 4, 1e-14, "mat_add 2×2");

    mat_sub(B, A, C, 2, 2);
    double expected_sub[4] = {4, 4, 4, 4};
    ASSERT_CLOSE(C, expected_sub, 4, 1e-14, "mat_sub 2×2");

    /* In-place: C = C + A  (C aliases first arg) */
    double D[4] = {1, 1, 1, 1};
    mat_add(D, A, D, 2, 2);
    double expected_inplace[4] = {2, 3, 4, 5};
    ASSERT_CLOSE(D, expected_inplace, 4, 1e-14, "mat_add in-place");
}


/* ================================================================== */
/* Test: mat_scale                                                      */
/* ================================================================== */

static void test_scale(void)
{
    printf("\n--- mat_scale ---\n");

    double A[4] = {1, 2, 3, 4};
    mat_scale(A, 2.5, 2, 2);
    double expected[4] = {2.5, 5.0, 7.5, 10.0};
    ASSERT_CLOSE(A, expected, 4, 1e-14, "mat_scale ×2.5");
}


/* ================================================================== */
/* Test: mat_mul                                                        */
/* ================================================================== */

static void test_mul(void)
{
    printf("\n--- mat_mul ---\n");

    /* 2×3 × 3×2 → 2×2 */
    double A[6] = {1, 2, 3,
                   4, 5, 6};
    double B[6] = {7, 8,
                   9, 10,
                   11, 12};
    double C[4];
    mat_mul(A, B, C, 2, 3, 2);
    /*
     * [ 1·7+2·9+3·11   1·8+2·10+3·12 ] = [  58,  64 ]
     * [ 4·7+5·9+6·11   4·8+5·10+6·12 ] = [ 139, 154 ]
     */
    double expected[4] = {58, 64, 139, 154};
    ASSERT_CLOSE(C, expected, 4, 1e-12, "mat_mul 2×3 × 3×2");

    /* 4×4 × identity = 4×4 */
    double F[16] = {
        1, 0.04, 8e-4,  2.133e-5,
        0, 1,    0.04,  8e-4,
        0, 0,    1,     0.04,
        0, 0,    0,     1
    };
    double I4[16];
    mat_eye(I4, 4);
    double FI[16];
    mat_mul(F, I4, FI, 4, 4, 4);
    ASSERT_CLOSE(F, FI, 16, 1e-14, "mat_mul F×I4 = F");

    /* 1×4 × 4×1 = 1×1 (inner product) */
    double H[4]  = {1, 0, 0, 0};
    double x[4]  = {1.75, 0.01, 0.002, 0.0};
    double Hx[1];
    mat_mul(H, x, Hx, 1, 4, 1);
    ASSERT_SCALAR(Hx[0], 1.75, 1e-14, "mat_mul H·x = C0");
}


/* ================================================================== */
/* Test: mat_transpose                                                  */
/* ================================================================== */

static void test_transpose(void)
{
    printf("\n--- mat_transpose ---\n");

    /* 2×3 → 3×2 */
    double A[6] = {1, 2, 3,
                   4, 5, 6};
    double At[6];
    mat_transpose(A, At, 2, 3);
    double expected[6] = {1, 4,
                           2, 5,
                           3, 6};
    ASSERT_CLOSE(At, expected, 6, 1e-15, "mat_transpose 2×3");

    /* Symmetric 3×3: Aᵀ = A */
    double S[9] = {1, 2, 3,
                   2, 5, 6,
                   3, 6, 9};
    double St[9];
    mat_transpose(S, St, 3, 3);
    ASSERT_CLOSE(S, St, 9, 1e-15, "mat_transpose symmetric");
}


/* ================================================================== */
/* Test: mat_inv                                                        */
/* ================================================================== */

static void test_inv(void)
{
    printf("\n--- mat_inv ---\n");

    /* 1×1 inverse: inv(3) = 1/3 */
    double A1[1] = {3.0};
    double Ainv1[1];
    int ret = mat_inv(A1, Ainv1, 1);
    ASSERT_INT_EQ(ret, 0, "mat_inv 1×1 return");
    ASSERT_SCALAR(Ainv1[0], 1.0/3.0, 1e-14, "mat_inv 1×1 value");

    /* 2×2 inverse:
     * A = [4, 7; 2, 6]  → det=4·6-7·2=10
     * A⁻¹ = (1/10)·[6, -7; -2, 4] = [0.6, -0.7; -0.2, 0.4]
     */
    double A2[4] = {4, 7, 2, 6};
    double Ainv2[4];
    ret = mat_inv(A2, Ainv2, 2);
    ASSERT_INT_EQ(ret, 0, "mat_inv 2×2 return");
    double expected2[4] = {0.6, -0.7, -0.2, 0.4};
    ASSERT_CLOSE(Ainv2, expected2, 4, 1e-13, "mat_inv 2×2 values");

    /* Verify A·A⁻¹ = I */
    double I2[4];
    mat_mul(A2, Ainv2, I2, 2, 2, 2);
    double Iref[4] = {1,0,0,1};
    ASSERT_CLOSE(I2, Iref, 4, 1e-13, "mat_inv A·A⁻¹ = I (2×2)");

    /* 3×3 inverse — verify A·A⁻¹ = I */
    double A3[9] = {
        2.0,  1.0,  0.0,
        1.0,  3.0,  1.0,
        0.0,  1.0,  2.0
    };
    double Ainv3[9];
    ret = mat_inv(A3, Ainv3, 3);
    ASSERT_INT_EQ(ret, 0, "mat_inv 3×3 return");

    double I3[9];
    mat_mul(A3, Ainv3, I3, 3, 3, 3);
    double Iref3[9] = {1,0,0, 0,1,0, 0,0,1};
    ASSERT_CLOSE(I3, Iref3, 9, 1e-13, "mat_inv A·A⁻¹ = I (3×3)");

    /* 4×4 identity inverse = identity */
    double I4[16], I4inv[16];
    mat_eye(I4, 4);
    ret = mat_inv(I4, I4inv, 4);
    ASSERT_INT_EQ(ret, 0, "mat_inv 4×4 I return");
    ASSERT_CLOSE(I4, I4inv, 16, 1e-13, "mat_inv I4⁻¹ = I4");

    /* Singular matrix should return -1 */
    double Asng[4] = {1, 2, 2, 4};   /* det = 0 */
    double Ainv_sng[4];
    ret = mat_inv(Asng, Ainv_sng, 2);
    ASSERT_INT_EQ(ret, -1, "mat_inv singular returns -1");
}


/* ================================================================== */
/* Test: mat_chol                                                       */
/* ================================================================== */

static void test_chol(void)
{
    printf("\n--- mat_chol ---\n");

    /* 3×3 SPD matrix, known answer:
     * A = [ 4,  2,  2 ]     L = [ 2,  0,    0   ]
     *     [ 2,  5,  3 ]         [ 1,  2,    0   ]
     *     [ 2,  3,  6 ]         [ 1,  1,  1.732 ]
     */
    double A[9] = {4, 2, 2,
                   2, 5, 3,
                   2, 3, 6};
    double L[9];
    int ret = mat_chol(A, L, 3);
    ASSERT_INT_EQ(ret, 0, "mat_chol return");
    ASSERT_SCALAR(L[0*3+0], 2.0,           1e-13, "L[0,0]");
    ASSERT_SCALAR(L[1*3+0], 1.0,           1e-13, "L[1,0]");
    ASSERT_SCALAR(L[1*3+1], 2.0,           1e-13, "L[1,1]");
    ASSERT_SCALAR(L[2*3+0], 1.0,           1e-13, "L[2,0]");
    ASSERT_SCALAR(L[2*3+1], 1.0,           1e-13, "L[2,1]");
    ASSERT_SCALAR(L[2*3+2], sqrt(4.0),     1e-13, "L[2,2]");   /* sqrt(6-1-1) = 2 */

    /* Verify L·Lᵀ = A */
    double Lt[9], LLt[9];
    mat_transpose(L, Lt, 3, 3);
    mat_mul(L, Lt, LLt, 3, 3, 3);
    ASSERT_CLOSE(A, LLt, 9, 1e-13, "mat_chol L·Lᵀ = A");

    /* Non-positive-definite should return -1 */
    double B[4] = {1, 0, 0, -1};   /* negative eigenvalue */
    double Lb[4];
    ret = mat_chol(B, Lb, 2);
    ASSERT_INT_EQ(ret, -1, "mat_chol non-PD returns -1");
}


/* ================================================================== */
/* Test: mat_symmetrise                                                 */
/* ================================================================== */

static void test_symmetrise(void)
{
    printf("\n--- mat_symmetrise ---\n");

    double A[9] = {1.0, 2.0, 3.0,
                   0.0, 5.0, 6.0,
                   0.0, 0.0, 9.0};
    mat_symmetrise(A, 3);
    double expected[9] = {1.0, 1.0, 1.5,
                           1.0, 5.0, 3.0,
                           1.5, 3.0, 9.0};
    ASSERT_CLOSE(A, expected, 9, 1e-15, "mat_symmetrise 3×3");
}


/* ================================================================== */
/* Test: KF-specific computation — one predict + update cycle         */
/* ================================================================== */

static void test_kf_cycle(void)
{
    printf("\n--- KF predict+update (manual) ---\n");

    /* Minimal 1-state KF: x=[p], F=1, H=1, Q=0.01, R=0.25
     * Equivalent to a scalar alpha-beta tracker.
     *
     * After predict from x=0, P=1:
     *   x_prior = 1·0 = 0
     *   P_prior = 1·1·1 + 0.01 = 1.01
     *
     * Update with z=1:
     *   y = 1 - 1·0 = 1
     *   S = 1·1.01·1 + 0.25 = 1.26
     *   K = 1.01 / 1.26 = 0.80158...
     *   x_post = 0 + 0.80158·1 = 0.80158
     *   P_post = (1 - 0.80158·1)·1.01·(1 - 0.80158·1)ᵀ
     *            + 0.80158·0.25·0.80158
     *          = 0.198·1.01·0.198 + 0.16063
     *          = 0.03970 + 0.16063 = 0.20032
     *   (Exact: P_post = R·P / (P+R) = 0.25·1.01/1.26 = 0.20040 — Joseph form matches)
     */
    double F[1]  = {1.0};
    double H[1]  = {1.0};
    double Q[1]  = {0.01};
    double R[1]  = {0.25};
    double x[1]  = {0.0};
    double P[1]  = {1.0};

    /* Predict */
    double x_new[1];
    mat_mul(F, x, x_new, 1, 1, 1);
    x[0] = x_new[0];

    double Ft[1], FP[1], FPFt[1], P_new[1];
    mat_transpose(F, Ft, 1, 1);
    mat_mul(F, P, FP, 1, 1, 1);
    mat_mul(FP, Ft, FPFt, 1, 1, 1);
    mat_add(FPFt, Q, P_new, 1, 1);
    P[0] = P_new[0];

    ASSERT_SCALAR(x[0], 0.0,  1e-14, "kf_cycle x after predict");
    ASSERT_SCALAR(P[0], 1.01, 1e-14, "kf_cycle P after predict");

    /* Update */
    double z[1]  = {1.0};
    double Hx[1]; mat_mul(H, x, Hx, 1, 1, 1);
    double y[1];  mat_sub(z, Hx, y, 1, 1);

    double S[1];
    double PHt[1]; mat_mul(P, H, PHt, 1, 1, 1);   /* H=Ht in 1D */
    double HPHt[1]; mat_mul(H, PHt, HPHt, 1, 1, 1);
    mat_add(HPHt, R, S, 1, 1);

    double Sinv[1]; mat_inv(S, Sinv, 1);
    double K[1]; mat_mul(PHt, Sinv, K, 1, 1, 1);

    double Ky[1]; mat_mul(K, y, Ky, 1, 1, 1);
    mat_add(x, Ky, x, 1, 1);

    /* Joseph form: P = (1-K)·P·(1-K) + K·R·K  (all scalars) */
    double IKH[1]  = {1.0 - K[0]};
    double IKH2[1] = {IKH[0] * IKH[0]};
    double lhs[1]  = {IKH2[0] * P[0]};
    double rhs[1]  = {K[0] * K[0] * R[0]};
    P[0] = lhs[0] + rhs[0];

    ASSERT_SCALAR(y[0], 1.0,    1e-14, "kf_cycle innovation");
    ASSERT_SCALAR(S[0], 1.26,   1e-12, "kf_cycle S");
    double K_expected = 1.01 / 1.26;
    ASSERT_SCALAR(K[0], K_expected, 1e-12, "kf_cycle K");
    ASSERT_SCALAR(x[0], K_expected, 1e-12, "kf_cycle x_post");

    /* P_post exact: R·P_prior/(P_prior+R) = 0.25·1.01/1.26 */
    double P_expected = (0.25 * 1.01) / 1.26;
    ASSERT_SCALAR(P[0], P_expected, 1e-12, "kf_cycle P_post (Joseph)");
}


/* ================================================================== */
/* main                                                                 */
/* ================================================================== */

int main(void)
{
    printf("=================================================\n");
    printf("  matrix.c unit tests\n");
    printf("=================================================\n");

    test_zero_eye();
    test_add_sub();
    test_scale();
    test_mul();
    test_transpose();
    test_inv();
    test_chol();
    test_symmetrise();
    test_kf_cycle();

    printf("\n=================================================\n");
    if (g_failures == 0)
        printf("  ALL TESTS PASSED\n");
    else
        printf("  %d TEST(S) FAILED\n", g_failures);
    printf("=================================================\n");

    return g_failures;   /* 0 = success for make/CI */
}
