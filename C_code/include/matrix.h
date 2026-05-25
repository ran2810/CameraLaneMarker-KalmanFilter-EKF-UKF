/**
 * matrix.h — Fixed-size matrix utilities for embedded Kalman filtering.
 *
 * Design philosophy
 * -----------------
 * - No heap allocation.  Every matrix lives on the caller's stack.
 * - Matrices are flat double arrays in ROW-MAJOR order.
 *   Element (i,j) of an [R x C] matrix M is  M[i*C + j].
 * - All functions write into a caller-supplied output buffer.
 *   No function returns a matrix by value.
 * - Dimensions are always explicit in the function name or signature
 *   so the compiler can inline and unroll loops at -O2.
 *
 * Naming convention
 * -----------------
 *   mat_<op>_NxM   — operates on a fixed N-row, M-col matrix
 *   mat_<op>       — generic version using runtime dimensions
 *
 * We define concrete fixed-size helpers for the sizes needed by the
 * lane-tracking KF (state n=4, obs m=1):
 *   4×4, 4×1, 1×4, 1×1
 *
 * Error handling
 * --------------
 * Functions that can fail (e.g. mat_inv — singular matrix) return int:
 *   0  = success
 *  -1  = failure (singular, dimension mismatch, etc.)
 * All other functions are void (they cannot fail for correct input).
 */

#ifndef MATRIX_H
#define MATRIX_H

#include <stddef.h>   /* size_t */
#include <stdio.h>    /* FILE* for mat_print */

/* ------------------------------------------------------------------ */
/* Convenience macros for index arithmetic                             */
/* ------------------------------------------------------------------ */

/** Element access: row i, col j of a matrix with C columns. */
#define MAT(m, C, i, j)  ((m)[(i)*(C) + (j)])

/** Shorthand for 4×4, 4×1, 1×4, 1×1 shapes used by the lane KF. */
#define M44(m, i, j)  MAT(m, 4, i, j)
#define M41(m, i)     ((m)[i])           /* column vector */
#define M14(m, j)     ((m)[j])           /* row vector    */
#define M11(m)        ((m)[0])


/* ------------------------------------------------------------------ */
/* Generic (runtime-dimension) operations                              */
/* ------------------------------------------------------------------ */

/**
 * mat_zero — fill an R×C matrix with zeros.
 */
void mat_zero(double *A, size_t R, size_t C);

/**
 * mat_eye — write the R×R identity matrix into A.
 * A must point to R*R doubles.
 */
void mat_eye(double *A, size_t R);

/**
 * mat_copy — deep copy of an R×C matrix: B = A.
 */
void mat_copy(const double *A, double *B, size_t R, size_t C);

/**
 * mat_add — element-wise addition: C = A + B.
 * All matrices are R×C.  C may alias A or B.
 */
void mat_add(const double *A, const double *B, double *C,
             size_t R, size_t Cols);

/**
 * mat_sub — element-wise subtraction: C = A − B.
 * All matrices are R×C.  C may alias A or B.
 */
void mat_sub(const double *A, const double *B, double *C,
             size_t R, size_t Cols);

/**
 * mat_scale — scalar multiply in-place: A *= s.
 */
void mat_scale(double *A, double s, size_t R, size_t C);

/**
 * mat_mul — general matrix multiply: C = A × B.
 *   A is [R × K],  B is [K × C_cols],  result C is [R × C_cols].
 * C must NOT alias A or B.
 */
void mat_mul(const double *A, const double *B, double *C,
             size_t R, size_t K, size_t C_cols);

/**
 * mat_transpose — C = Aᵀ.
 *   A is [R × C],  result is [C × R].
 * C must NOT alias A.
 */
void mat_transpose(const double *A, double *C, size_t R, size_t Cols);

/**
 * mat_inv — square matrix inverse via LU decomposition with partial pivoting.
 *   Overwrites Ainv with A⁻¹.  A is [N × N], Ainv is [N × N].
 *   A is NOT modified (a working copy is made internally on the stack;
 *   max supported N is MATRIX_MAX_DIM).
 *
 * Returns:
 *   0   success
 *  -1   singular matrix (pivot < MATRIX_EPS)
 */
int mat_inv(const double *A, double *Ainv, size_t N);

/**
 * mat_chol — Cholesky decomposition: A = L·Lᵀ for symmetric positive-definite A.
 *   Writes lower-triangular L into out (upper triangle is zeroed).
 *   A is [N × N].
 *
 * Returns:
 *   0   success
 *  -1   matrix is not positive-definite
 */
int mat_chol(const double *A, double *L, size_t N);

/**
 * mat_symmetrise — enforce symmetry: A = 0.5*(A + Aᵀ) in-place.
 * Used to prevent covariance drift from floating-point rounding.
 */
void mat_symmetrise(double *A, size_t N);

/**
 * mat_print — formatted print to a FILE (pass stdout for console).
 *   label is printed above the matrix.
 */
void mat_print(const double *A, size_t R, size_t C,
               const char *label, FILE *f);


/* ------------------------------------------------------------------ */
/* Configuration                                                       */
/* ------------------------------------------------------------------ */

/**
 * Maximum matrix dimension supported by mat_inv and mat_chol.
 * Stack-allocated working buffers are sized to this.
 * Increase if you need larger state vectors.
 */
#define MATRIX_MAX_DIM  8

/**
 * Pivot threshold below which a matrix is considered singular.
 */
#define MATRIX_EPS      1e-12


#endif /* MATRIX_H */
