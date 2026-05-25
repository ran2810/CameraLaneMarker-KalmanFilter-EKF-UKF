/**
 * matrix.c — Fixed-size matrix utilities for embedded Kalman filtering.
 *
 * See matrix.h for the public API and design notes.
 *
 * Implementation notes
 * --------------------
 * All matrices are stored row-major in flat double arrays.
 * No dynamic allocation anywhere in this file.
 * LU decomposition uses partial (row) pivoting for numerical stability.
 */

#include "matrix.h"

#include <math.h>    /* fabs, sqrt */
#include <string.h>  /* memcpy, memset */
#include <stdio.h>


/* ================================================================== */
/* Basic operations                                                    */
/* ================================================================== */

void mat_zero(double *A, size_t R, size_t C)
{
    memset(A, 0, R * C * sizeof(double));
}

void mat_eye(double *A, size_t R)
{
    mat_zero(A, R, R);
    for (size_t i = 0; i < R; i++)
        A[i * R + i] = 1.0;
}

void mat_copy(const double *A, double *B, size_t R, size_t C)
{
    memcpy(B, A, R * C * sizeof(double));
}

void mat_add(const double *A, const double *B, double *C,
             size_t R, size_t Cols)
{
    size_t n = R * Cols;
    for (size_t i = 0; i < n; i++)
        C[i] = A[i] + B[i];
}

void mat_sub(const double *A, const double *B, double *C,
             size_t R, size_t Cols)
{
    size_t n = R * Cols;
    for (size_t i = 0; i < n; i++)
        C[i] = A[i] - B[i];
}

void mat_scale(double *A, double s, size_t R, size_t C)
{
    size_t n = R * C;
    for (size_t i = 0; i < n; i++)
        A[i] *= s;
}


/* ================================================================== */
/* Matrix multiply: C = A [R×K] × B [K×Ccols]                        */
/* ================================================================== */

void mat_mul(const double *A, const double *B, double *C,
             size_t R, size_t K, size_t C_cols)
{
    /* C must not alias A or B — caller's responsibility */
    for (size_t i = 0; i < R; i++) {
        for (size_t j = 0; j < C_cols; j++) {
            double acc = 0.0;
            for (size_t k = 0; k < K; k++)
                acc += A[i * K + k] * B[k * C_cols + j];
            C[i * C_cols + j] = acc;
        }
    }
}


/* ================================================================== */
/* Transpose: C [Ccols×R] = Aᵀ [R×Ccols]                             */
/* ================================================================== */

void mat_transpose(const double *A, double *C, size_t R, size_t Cols)
{
    /* C must not alias A */
    for (size_t i = 0; i < R; i++)
        for (size_t j = 0; j < Cols; j++)
            C[j * R + i] = A[i * Cols + j];
}


/* ================================================================== */
/* LU decomposition with partial pivoting (in-place on a copy)        */
/*                                                                     */
/* Decomposes  A  →  P·A = L·U  where:                               */
/*   piv[i] = index of the row swapped into position i               */
/*   L is unit lower-triangular (1s on diagonal, stored below)        */
/*   U is upper-triangular (stored on and above diagonal)             */
/*                                                                     */
/* Returns 0 on success, -1 if a zero pivot is encountered.           */
/* ================================================================== */

static int lu_decompose(double *LU,       /* N×N, modified in-place  */
                        size_t *piv,      /* pivot indices [N]        */
                        size_t N)
{
    for (size_t k = 0; k < N; k++) {
        /* Find the largest pivot in column k at or below row k */
        size_t p = k;
        double max_val = fabs(LU[k * N + k]);
        for (size_t i = k + 1; i < N; i++) {
            double v = fabs(LU[i * N + k]);
            if (v > max_val) { max_val = v; p = i; }
        }

        if (max_val < MATRIX_EPS)
            return -1;   /* singular */

        piv[k] = p;

        /* Swap rows k and p */
        if (p != k) {
            for (size_t j = 0; j < N; j++) {
                double tmp      = LU[k * N + j];
                LU[k * N + j]  = LU[p * N + j];
                LU[p * N + j]  = tmp;
            }
        }

        /* Eliminate below the pivot */
        double pivot = LU[k * N + k];
        for (size_t i = k + 1; i < N; i++) {
            LU[i * N + k] /= pivot;                        /* L factor */
            for (size_t j = k + 1; j < N; j++)
                LU[i * N + j] -= LU[i * N + k] * LU[k * N + j]; /* U factor */
        }
    }
    return 0;
}

/*
 * Solve LU·x = Pb  given the LU factorisation and pivot array.
 * Writes solution into x [N].
 */
static void lu_solve(const double *LU, const size_t *piv,
                     const double *b,  double *x, size_t N)
{
    /* Apply row permutation to b → xp */
    double xp[MATRIX_MAX_DIM];
    for (size_t i = 0; i < N; i++)
        xp[i] = b[i];
    for (size_t k = 0; k < N; k++) {
        double tmp  = xp[k];
        xp[k]       = xp[piv[k]];
        xp[piv[k]]  = tmp;
    }

    /* Forward substitution  L·y = Pb  (L has unit diagonal) */
    for (size_t i = 0; i < N; i++) {
        x[i] = xp[i];
        for (size_t j = 0; j < i; j++)
            x[i] -= LU[i * N + j] * x[j];
    }

    /* Back substitution  U·x = y */
    for (size_t i = N; i-- > 0; ) {
        for (size_t j = i + 1; j < N; j++)
            x[i] -= LU[i * N + j] * x[j];
        x[i] /= LU[i * N + i];
    }
}


/* ================================================================== */
/* Matrix inverse via LU  (A is NOT modified)                         */
/* ================================================================== */

int mat_inv(const double *A, double *Ainv, size_t N)
{
    /* Stack-allocated working copy of A */
    double LU[MATRIX_MAX_DIM * MATRIX_MAX_DIM];
    size_t piv[MATRIX_MAX_DIM];
    double col[MATRIX_MAX_DIM];   /* unit basis vector */
    double sol[MATRIX_MAX_DIM];   /* solution column   */

    mat_copy(A, LU, N, N);

    if (lu_decompose(LU, piv, N) != 0)
        return -1;   /* singular */

    /* Invert column by column: solve A·X = I */
    for (size_t j = 0; j < N; j++) {
        /* Build j-th unit basis vector */
        for (size_t i = 0; i < N; i++) col[i] = (i == j) ? 1.0 : 0.0;

        lu_solve(LU, piv, col, sol, N);

        /* Write solution into j-th column of Ainv */
        for (size_t i = 0; i < N; i++)
            Ainv[i * N + j] = sol[i];
    }
    return 0;
}


/* ================================================================== */
/* Cholesky decomposition: A = L·Lᵀ  (A symmetric positive-definite) */
/* ================================================================== */

int mat_chol(const double *A, double *L, size_t N)
{
    mat_zero(L, N, N);

    for (size_t i = 0; i < N; i++) {
        for (size_t j = 0; j <= i; j++) {
            double s = A[i * N + j];
            for (size_t k = 0; k < j; k++)
                s -= L[i * N + k] * L[j * N + k];

            if (i == j) {
                if (s < 0.0) return -1;   /* not positive-definite */
                L[i * N + j] = sqrt(s);
            } else {
                if (fabs(L[j * N + j]) < MATRIX_EPS) return -1;
                L[i * N + j] = s / L[j * N + j];
            }
        }
    }
    return 0;
}


/* ================================================================== */
/* Symmetrise: A = 0.5*(A + Aᵀ)  in-place                            */
/* ================================================================== */

void mat_symmetrise(double *A, size_t N)
{
    for (size_t i = 0; i < N; i++) {
        for (size_t j = i + 1; j < N; j++) {
            double avg      = 0.5 * (A[i * N + j] + A[j * N + i]);
            A[i * N + j]    = avg;
            A[j * N + i]    = avg;
        }
    }
}


/* ================================================================== */
/* Debug print                                                         */
/* ================================================================== */

void mat_print(const double *A, size_t R, size_t C,
               const char *label, FILE *f)
{
    if (label) fprintf(f, "%s  [%zu × %zu]\n", label, R, C);
    for (size_t i = 0; i < R; i++) {
        fprintf(f, "  [");
        for (size_t j = 0; j < C; j++)
            fprintf(f, " %12.6f", A[i * C + j]);
        fprintf(f, " ]\n");
    }
    fprintf(f, "\n");
}
