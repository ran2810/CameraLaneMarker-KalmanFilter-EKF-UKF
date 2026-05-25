/**
 * KalmanFilter.hpp — Linear Kalman Filter (header-only).
 *
 * Model
 * -----
 *   x_{k+1} = F · x_k + w_k,    w_k ~ N(0, Q)
 *   z_k     = H · x_k + v_k,    v_k ~ N(0, R)
 *
 * Equations
 * ---------
 *   Predict:
 *     x̂⁻ = F · x̂
 *     P⁻  = F · P · Fᵀ + Q
 *
 *   Update:
 *     y  = z − H · x̂⁻
 *     S  = H · P⁻ · Hᵀ + R
 *     K  = P⁻ · Hᵀ · S⁻¹
 *     x̂  = x̂⁻ + K · y
 *     P  = (I−KH)·P⁻·(I−KH)ᵀ + K·R·Kᵀ   ← Joseph form
 *
 * Usage (lane tracking)
 * ---------------------
 *   auto kf = KalmanFilter<4,1>::make_lane_filter(dt, sp, sigma_camera);
 *   kf.initialize(x0, P0);
 *
 *   for each frame:
 *       kf.predict();
 *       if (meas_valid)
 *           kf.update(z);
 *       double C0_hat = kf.x(0);
 *
 * No heap allocation for N,M ≤ ~16.  Eigen fixed-size matrices are
 * stored inline in the object (stack-friendly).
 */

#pragma once

#include "FilterBase.hpp"
#include "LaneModel.hpp"

namespace kf {

// ---------------------------------------------------------------------------
// KalmanFilter<N, M>
// ---------------------------------------------------------------------------

template <int N, int M>
class KalmanFilter : public FilterBase<KalmanFilter<N,M>, N, M> {
public:
    using Base      = FilterBase<KalmanFilter<N,M>, N, M>;
    using StateVec  = typename Base::StateVec;
    using StateMat  = typename Base::StateMat;
    using ObsVec    = typename Base::ObsVec;
    using HMat      = typename Base::HMat;

    // Model matrices — set by factory or directly
    StateMat F = StateMat::Identity();   ///< state transition
    HMat     H = HMat::Zero();           ///< observation matrix

    const char* filter_name() const { return "KF"; }

    // ------------------------------------------------------------------
    // Factory: lane-tracking KF
    // ------------------------------------------------------------------
    /**
     * Build a ready-to-use lane-tracking KF.
     *
     * @param dt            Camera frame period (s), e.g. 0.04 for 25 Hz.
     * @param sp            Process noise std-devs (C0, C1, C2, C3).
     * @param sigma_camera  Std-dev of camera C0 measurement noise (m).
     * @return              Configured KalmanFilter.
     */
    static KalmanFilter make_lane_filter(double              dt,
                                         const SigmaProcess& sp,
                                         double              sigma_camera)
    {
        KalmanFilter kf;
        kf.F    = lane::make_F(dt);
        kf.H    = lane::make_H();
        kf.Q    = lane::make_Q(sp.C0, sp.C1, sp.C2, sp.C3);
        kf.R    = lane::make_R(sigma_camera);
        kf.P    = lane::make_P0(sp.C0, sp.C1, sp.C2, sp.C3);
        return kf;
    }

    // ------------------------------------------------------------------
    // CRTP predict implementation
    // ------------------------------------------------------------------
    /**
     * Time update (no control input needed for the linear model).
     *
     *   x̂⁻ = F · x̂
     *   P⁻  = F · P · Fᵀ + Q
     *
     * @return  Predicted state x̂⁻.
     */
    const StateVec& predict_impl()
    {
        this->x = F * this->x;
        this->P = F * this->P * F.transpose() + this->Q;
        return this->x;
    }

    // ------------------------------------------------------------------
    // CRTP update implementation
    // ------------------------------------------------------------------
    /**
     * Measurement update.
     *
     * @param z  Observation [M].
     * @return   true on success, false if S is singular.
     */
    bool update_impl(const ObsVec& z)
    {
        const ObsVec z_pred = H * this->x;
        return this->measurement_update_kernel(z, z_pred, H);
    }
};

// Convenience alias for the lane-tracking instance
using LaneKF = KalmanFilter<lane::N, lane::M>;

} // namespace kf
