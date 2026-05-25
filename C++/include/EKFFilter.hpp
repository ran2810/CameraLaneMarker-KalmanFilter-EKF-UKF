/**
 * EKFFilter.hpp — Extended Kalman Filter (header-only).
 *
 * Model
 * -----
 *   x_{k+1} = f(x_k, u_k) + w_k,    w_k ~ N(0, Q)
 *   z_k     = h(x_k)      + v_k,    v_k ~ N(0, R)
 *
 * The EKF linearises f around the current estimate using the Jacobian
 * F_jac = ∂f/∂x, and h around the current estimate using H_jac = ∂h/∂x.
 *
 * Equations
 * ---------
 *   Predict:
 *     x̂⁻ = f(x̂, u, dt)
 *     F   = ∂f/∂x |_{x̂, u}
 *     P⁻  = F · P · Fᵀ + Q
 *
 *   Update  (same kernel as KF, just with different z_pred and H_jac):
 *     z_pred = h(x̂⁻)
 *     H      = ∂h/∂x |_{x̂⁻}
 *     y      = z − z_pred
 *     S      = H·P⁻·Hᵀ + R
 *     K      = P⁻·Hᵀ·S⁻¹
 *     x̂      = x̂⁻ + K·y
 *     P      = (I−KH)·P⁻·(I−KH)ᵀ + K·R·Kᵀ
 *
 * For the lane-tracking application:
 *   f and F_jac are defined in LaneModel.hpp (ego-motion compensation).
 *   h(x) = C0 = x[0]  →  H_jac = [1, 0, 0, 0]  (same as linear case).
 *
 * The f/h/F_jac/H_jac functions are stored as std::function so the class
 * can be used with arbitrary user-supplied models without subclassing.
 *
 * Usage (lane tracking)
 * ---------------------
 *   auto ekf = EKFFilter<4,1,2>::make_lane_filter(dt, sp, sigma_camera);
 *   ekf.initialize(x0, P0);
 *
 *   for each frame:
 *       ekf.predict(u, dt);    // u = [speed, yaw_rate]
 *       if (meas_valid)
 *           ekf.update(z);
 */

#pragma once

#include "FilterBase.hpp"
#include "LaneModel.hpp"

#include <functional>

namespace kf {

// ---------------------------------------------------------------------------
// EKFFilter<N, M, U>
//   N = state dim,  M = obs dim,  U = control dim
// ---------------------------------------------------------------------------

template <int N, int M, int U>
class EKFFilter : public FilterBase<EKFFilter<N,M,U>, N, M> {
public:
    using Base      = FilterBase<EKFFilter<N,M,U>, N, M>;
    using StateVec  = typename Base::StateVec;
    using StateMat  = typename Base::StateMat;
    using ObsVec    = typename Base::ObsVec;
    using HMat      = typename Base::HMat;
    using CtrlVec   = Eigen::Matrix<double, U, 1>;

    // ------------------------------------------------------------------
    // Nonlinear functions — caller-supplied (set by factory or directly)
    // ------------------------------------------------------------------

    /// Nonlinear state transition:  f(x, u, dt) → x_new
    std::function<StateVec(const StateVec&, const CtrlVec&, double)> f_func;

    /// Jacobian of f w.r.t. x:  F_jac(x, u, dt) → [N×N]
    std::function<StateMat(const StateVec&, const CtrlVec&, double)> F_jac_func;

    /// Measurement function:  h(x) → z_pred  [M]
    std::function<ObsVec(const StateVec&)> h_func;

    /// Jacobian of h w.r.t. x:  H_jac(x) → [M×N]
    std::function<HMat(const StateVec&)>   H_jac_func;

    const char* filter_name() const { return "EKF"; }

    // ------------------------------------------------------------------
    // Factory: lane-tracking EKF
    // ------------------------------------------------------------------
    static EKFFilter make_lane_filter(double              dt,
                                      const SigmaProcess& sp,
                                      double              sigma_camera)
    {
        (void)dt;   // dt passed per-step, not cached
        EKFFilter ekf;

        ekf.Q = lane::make_Q(sp.C0, sp.C1, sp.C2, sp.C3);
        ekf.R = lane::make_R(sigma_camera);
        ekf.P = lane::make_P0(sp.C0, sp.C1, sp.C2, sp.C3);

        // Nonlinear state transition (ego-motion compensation)
        ekf.f_func = [](const StateVec& x, const CtrlVec& u, double dt_) {
            return lane::f(x, u, dt_);
        };

        // Jacobian ∂f/∂x
        ekf.F_jac_func = [](const StateVec& x, const CtrlVec& u, double dt_) {
            return lane::F_jac(x, u, dt_);
        };

        // Measurement: h(x) = C0 = x[0]
        ekf.h_func = [](const StateVec& x) {
            return lane::h(x);
        };

        // H_jac = [1, 0, 0, 0]  (constant, same as linear H)
        ekf.H_jac_func = [](const StateVec&) {
            return lane::make_H();
        };

        return ekf;
    }

    // ------------------------------------------------------------------
    // CRTP predict implementation
    // ------------------------------------------------------------------
    /**
     * Nonlinear time update.
     *
     *   x̂⁻ = f(x̂, u, dt)
     *   F   = ∂f/∂x |_{x̂, u}       (Jacobian)
     *   P⁻  = F · P · Fᵀ + Q
     *
     * @param u   Control vector [v, ω] (speed m/s, yaw rate rad/s).
     * @param dt  Sample period (s).
     * @return    Predicted state x̂⁻.
     */
    const StateVec& predict_impl(const CtrlVec& u, double dt)
    {
        // Propagate mean through nonlinear f
        this->x = f_func(this->x, u, dt);

        // Linearise at the new estimate
        const StateMat Fj = F_jac_func(this->x, u, dt);

        // Propagate covariance through linearisation
        this->P = Fj * this->P * Fj.transpose() + this->Q;

        return this->x;
    }

    // ------------------------------------------------------------------
    // CRTP update implementation
    // ------------------------------------------------------------------
    /**
     * Measurement update using linearised h(x).
     *
     * @param z  Observation [M].
     * @return   true on success, false if S is singular.
     */
    bool update_impl(const ObsVec& z)
    {
        const ObsVec z_pred = h_func(this->x);
        const HMat   Hj     = H_jac_func(this->x);
        return this->measurement_update_kernel(z, z_pred, Hj);
    }
};

// Convenience alias
using LaneEKF = EKFFilter<lane::N, lane::M, lane::U>;

} // namespace kf
