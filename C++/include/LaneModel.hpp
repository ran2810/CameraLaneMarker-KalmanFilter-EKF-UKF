/**
 * LaneModel.hpp — Lane-marker state-space model shared by all three filters.
 *
 * State vector  x = [C0, C1, C2, C3]ᵀ   (N = 4)
 *   C0  lateral offset to lane marking (m)
 *   C1  heading angle relative to lane (rad)
 *   C2  lane curvature (1/m)
 *   C3  curvature rate (1/m²)
 *
 * Observation   z = [C0]                 (M = 1)
 *   Camera reports the lateral offset to a lane marking.
 *
 * Linear process model — KF (small-angle approximation, sin C1 ≈ C1):
 *
 *   F(dt) = [[1,  dt,  dt²/2,  dt³/6],
 *            [0,   1,     dt,  dt²/2],
 *            [0,   0,      1,     dt],
 *            [0,   0,      0,      1]]
 *
 * Nonlinear process model — EKF / UKF  (ego-motion compensation):
 *
 *   f₁ = C0 − v·dt·sin(C1)
 *   f₂ = C1 + C2·v·dt − ω·dt
 *   f₃ = C2 + C3·v·dt
 *   f₄ = C3
 *
 * EKF Jacobian  ∂f/∂x  (4×4):
 *
 *   [[1, −v·dt·cos(C1), 0,      0   ],
 *    [0,      1,        v·dt,   0   ],
 *    [0,      0,         1,    v·dt ],
 *    [0,      0,         0,     1   ]]
 *
 * Measurement model  (linear, same for all three filters):
 *   h(x) = C0 = x[0]     →   H = [1, 0, 0, 0]
 *
 * Control vector  u = [v, ω]   vehicle speed (m/s), yaw rate (rad/s).
 */

#pragma once

#include <Eigen/Dense>
#include <cmath>

namespace kf {
namespace lane {

// ---------------------------------------------------------------------------
// Compile-time dimensions
// ---------------------------------------------------------------------------
static constexpr int N = 4;   ///< state dimension
static constexpr int M = 1;   ///< observation dimension
static constexpr int U = 2;   ///< control dimension [v, ω]

// ---------------------------------------------------------------------------
// Eigen type aliases for lane-tracking sizes
// ---------------------------------------------------------------------------
using StateVec = Eigen::Matrix<double, N, 1>;
using StateMat = Eigen::Matrix<double, N, N>;
using ObsVec   = Eigen::Matrix<double, M, 1>;
using ObsMat   = Eigen::Matrix<double, M, M>;
using CtrlVec  = Eigen::Matrix<double, U, 1>;
using HMat     = Eigen::Matrix<double, M, N>;

// ---------------------------------------------------------------------------
// Linear state-transition matrix F(dt) — clothoid kinematic model
//
//   Derived from Taylor expansion of the clothoid (Euler spiral) road model
//   under constant curvature-rate assumption.
// ---------------------------------------------------------------------------
inline StateMat make_F(double dt) noexcept
{
    const double dt2 = dt * dt;
    const double dt3 = dt * dt * dt;
    StateMat F;
    F << 1.0,  dt,  0.5*dt2,  dt3/6.0,
         0.0, 1.0,       dt,  0.5*dt2,
         0.0, 0.0,      1.0,       dt,
         0.0, 0.0,      0.0,      1.0;
    return F;
}

// ---------------------------------------------------------------------------
// Observation matrix H = [1, 0, 0, 0]
// ---------------------------------------------------------------------------
inline HMat make_H() noexcept
{
    HMat H = HMat::Zero();
    H(0, 0) = 1.0;
    return H;
}

// ---------------------------------------------------------------------------
// Process noise covariance Q = diag(σ²_C0, σ²_C1, σ²_C2, σ²_C3)
// ---------------------------------------------------------------------------
inline StateMat make_Q(double sig_C0, double sig_C1,
                        double sig_C2, double sig_C3) noexcept
{
    StateMat Q = StateMat::Zero();
    Q(0,0) = sig_C0 * sig_C0;
    Q(1,1) = sig_C1 * sig_C1;
    Q(2,2) = sig_C2 * sig_C2;
    Q(3,3) = sig_C3 * sig_C3;
    return Q;
}

// ---------------------------------------------------------------------------
// Measurement noise covariance R = [σ²_camera]
// ---------------------------------------------------------------------------
inline ObsMat make_R(double sig_camera) noexcept
{
    ObsMat R;
    R(0,0) = sig_camera * sig_camera;
    return R;
}

// ---------------------------------------------------------------------------
// Initial covariance P0 — loose diagonal prior
//   Default: P0 = 10 × Q diagonal
// ---------------------------------------------------------------------------
inline StateMat make_P0(double sig_C0, double sig_C1,
                         double sig_C2, double sig_C3,
                         double scale = 10.0) noexcept
{
    StateMat P = StateMat::Zero();
    P(0,0) = scale * sig_C0 * sig_C0;
    P(1,1) = scale * sig_C1 * sig_C1;
    P(2,2) = scale * sig_C2 * sig_C2;
    P(3,3) = scale * sig_C3 * sig_C3;
    return P;
}

// ---------------------------------------------------------------------------
// Nonlinear state transition  f(x, u, dt)
//   Used by EKF and UKF.  The KF uses the linearised F(dt) above.
// ---------------------------------------------------------------------------
inline StateVec f(const StateVec& x, const CtrlVec& u, double dt) noexcept
{
    const double C0    = x(0);
    const double C1    = x(1);
    const double C2    = x(2);
    const double C3    = x(3);
    const double v     = u(0);
    const double omega = u(1);

    StateVec xn;
    xn(0) = C0 - v * dt * std::sin(C1);       // lateral ego-motion
    xn(1) = C1 + C2 * v * dt - omega * dt;    // heading update
    xn(2) = C2 + C3 * v * dt;                 // curvature update
    xn(3) = C3;                                // curvature rate constant
    return xn;
}

// ---------------------------------------------------------------------------
// EKF Jacobian  F_jac = ∂f/∂x  evaluated at current state x
// ---------------------------------------------------------------------------
inline StateMat F_jac(const StateVec& x, const CtrlVec& u, double dt) noexcept
{
    const double C1    = x(1);
    const double v     = u(0);

    StateMat Fj;
    Fj << 1.0, -v * dt * std::cos(C1),    0.0,      0.0,
          0.0,                      1.0,  v * dt,    0.0,
          0.0,                      0.0,     1.0,  v * dt,
          0.0,                      0.0,     0.0,     1.0;
    return Fj;
}

// ---------------------------------------------------------------------------
// Measurement function  h(x) = C0 = x[0]
// ---------------------------------------------------------------------------
inline ObsVec h(const StateVec& x) noexcept
{
    return x.template head<M>();   // [C0]
}

} // namespace lane
} // namespace kf
