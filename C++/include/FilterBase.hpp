/**
 * FilterBase.hpp — CRTP abstract base class for all Kalman filter variants.
 *
 * Design notes
 * ------------
 * Uses CRTP (Curiously Recurring Template Pattern) so that predict() and
 * update() calls dispatch at compile time with zero virtual-function overhead
 * while still sharing the common state (x, P, Q, R, diagnostics).
 *
 * Template parameters
 *   Derived   — the concrete filter type (KalmanFilter, EKFFilter, ...)
 *   N         — state dimension  (compile-time constant)
 *   M         — observation dimension (compile-time constant)
 *
 * Eigen type aliases expose the right fixed-size matrix types to subclasses:
 *   StateVec   — Vector<double, N>
 *   StateMat   — Matrix<double, N, N>
 *   ObsVec     — Vector<double, M>
 *   ObsMat     — Matrix<double, M, M>
 *   GainMat    — Matrix<double, N, M>
 *   HMat       — Matrix<double, M, N>
 *
 * Eigen fixed-size matrices are stack-allocated for N,M ≤ ~16.
 * For larger state vectors, replace with Dynamic matrices.
 *
 * NIS (Normalised Innovation Squared)
 *   Updated by update() each cycle.  Should follow χ²(M) if the filter
 *   is statistically consistent.  E[NIS] = M.
 *
 * Matching the C interface
 *   - SigmaProcess struct (same field names as C version)
 *   - Same predict/update split
 *   - Same Joseph-form covariance update for numerical stability
 *   - Same filter parameters and defaults
 */

#pragma once

#include <Eigen/Dense>
#include <iostream>
#include <iomanip>

namespace kf {

// ---------------------------------------------------------------------------
// Helper: process-noise parameter bundle (mirrors C SigmaProcess)
// ---------------------------------------------------------------------------

struct SigmaProcess {
    double C0 = 0.07;    ///< std-dev of C0 process noise (m)
    double C1 = 0.004;   ///< std-dev of C1 process noise (rad)
    double C2 = 4e-5;    ///< std-dev of C2 process noise (1/m)
    double C3 = 8e-7;    ///< std-dev of C3 process noise (1/m²)
};


// ---------------------------------------------------------------------------
// FilterBase<Derived, N, M>
// ---------------------------------------------------------------------------

template <typename Derived, int N, int M>
class FilterBase {
public:
    // ------------------------------------------------------------------
    // Eigen type aliases used throughout the filter hierarchy
    // ------------------------------------------------------------------
    using StateVec = Eigen::Matrix<double, N, 1>;
    using StateMat = Eigen::Matrix<double, N, N>;
    using ObsVec   = Eigen::Matrix<double, M, 1>;
    using ObsMat   = Eigen::Matrix<double, M, M>;
    using GainMat  = Eigen::Matrix<double, N, M>;
    using HMat     = Eigen::Matrix<double, M, N>;

    static constexpr int StateSize = N;
    static constexpr int ObsSize   = M;

    // ------------------------------------------------------------------
    // State — accessible from subclasses and examples
    // ------------------------------------------------------------------
    StateVec x  = StateVec::Zero();   ///< posterior state estimate x̂
    StateMat P  = StateMat::Zero();   ///< posterior covariance
    StateMat Q  = StateMat::Zero();   ///< process noise covariance
    ObsMat   R  = ObsMat::Zero();     ///< measurement noise covariance

    // Diagnostics — written each update() call
    ObsVec   innovation = ObsVec::Zero();  ///< y = z − h(x̂⁻)
    ObsMat   S          = ObsMat::Zero();  ///< innovation covariance
    GainMat  K          = GainMat::Zero(); ///< Kalman gain
    double   NIS        = 0.0;            ///< Normalised Innovation Squared

    // ------------------------------------------------------------------
    // CRTP dispatch — the public interface
    // ------------------------------------------------------------------

    /**
     * Initialize state and covariance.
     * @param x0    Initial state estimate.
     * @param P0    Initial covariance.
     */
    void initialize(const StateVec& x0, const StateMat& P0) {
        x = x0;
        P = P0;
    }

    /**
     * Time-update step.  Signature depends on the concrete filter:
     *   KF:  predict()
     *   EKF: predict(u, dt)
     *   UKF: predict(u, dt)
     * Delegates to Derived::predict_impl via CRTP.
     */
    template <typename... Args>
    const StateVec& predict(Args&&... args) {
        return derived().predict_impl(std::forward<Args>(args)...);
    }

    /**
     * Measurement-update step.
     * @param z  Observation vector [M].
     * @return   true on success, false if S is singular.
     */
    bool update(const ObsVec& z) {
        return derived().update_impl(z);
    }

    // ------------------------------------------------------------------
    // Shared update kernel — called by every concrete update_impl()
    //
    //   innovation = z − z_pred
    //   S          = HPHt + R
    //   K          = PHt · S⁻¹
    //   x          = x + K · innovation
    //   P          = (I-KH)·P·(I-KH)ᵀ + K·R·Kᵀ   ← Joseph form
    //   NIS        = innovᵀ · S⁻¹ · innov
    //
    // Templated on Hjac type so both the linear HMat and an ad-hoc
    // Eigen expression (e.g. from the EKF Jacobian) work transparently.
    // ------------------------------------------------------------------
    template <typename HType>
    bool measurement_update_kernel(const ObsVec&  z,
                                   const ObsVec&  z_pred,
                                   const HType&   H_jac)
    {
        innovation = z - z_pred;

        // S = H·P·Hᵀ + R
        S = H_jac * P * H_jac.transpose() + R;

        // Solve for K = P·Hᵀ·S⁻¹  using LDLT factorisation (faster & stable)
        Eigen::LDLT<ObsMat> ldlt(S);
        if (ldlt.info() != Eigen::Success) return false;

        K = ldlt.solve(H_jac * P).transpose();   // K = (S⁻¹ · H · P)ᵀ

        // State update
        x = x + K * innovation;

        // Joseph-form covariance update — numerically stable, keeps P ≥ 0
        StateMat IKH = StateMat::Identity() - K * H_jac;
        P = IKH * P * IKH.transpose() + K * R * K.transpose();

        // Enforce exact symmetry
        P = (P + P.transpose()) * 0.5;

        // NIS
        NIS = (innovation.transpose() * ldlt.solve(innovation)).value();

        return true;
    }

    // ------------------------------------------------------------------
    // Pretty print
    // ------------------------------------------------------------------
    void print(std::ostream& os = std::cout) const {
        os << derived().filter_name() << " State:\n"
           << std::fixed << std::setprecision(6)
           << "  x̂  = [C0=" << x(0)
           << "  C1=" << x(1)
           << "  C2=" << x(2)
           << "  C3=" << x(3) << "]\n"
           << "  P diag = ["
           << P(0,0) << "  " << P(1,1) << "  "
           << P(2,2) << "  " << P(3,3) << "]\n"
           << "  NIS=" << NIS
           << "  innov=" << innovation(0) << "\n";
    }

protected:
    // ------------------------------------------------------------------
    // CRTP helpers
    // ------------------------------------------------------------------
    Derived&       derived()       { return static_cast<Derived&>(*this); }
    const Derived& derived() const { return static_cast<const Derived&>(*this); }

    // Default filter name — overridden by subclasses
    const char* filter_name() const { return "Filter"; }
};

} // namespace kf
