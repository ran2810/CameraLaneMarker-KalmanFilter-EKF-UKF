/**
 * UKFFilter.hpp — Unscented Kalman Filter (header-only).
 *
 * Uses the Van der Merwe scaled sigma-point transform to propagate
 * the Gaussian through the nonlinear functions f and h.
 * No Jacobians required — only the functions themselves.
 *
 * Sigma-point parameters (α, β, κ)
 * ---------------------------------
 *   α   — spread around mean        (typical: 1e-3)
 *   β   — prior distribution info   (β=2 optimal for Gaussian)
 *   κ   — secondary scaling          (typical: 0)
 *   λ   = α²(n+κ) − n
 *
 * 2n+1 sigma points (n = state dim):
 *   X₀     = x̂
 *   Xᵢ     = x̂ + (√((n+λ)P))ᵢ      i = 1…n
 *   Xᵢ₊ₙ   = x̂ − (√((n+λ)P))ᵢ      i = 1…n
 *
 * Weights:
 *   Wm[0] = λ/(n+λ)
 *   Wc[0] = λ/(n+λ) + (1 − α² + β)
 *   Wm[i] = Wc[i] = 1/(2(n+λ))       for i = 1…2n
 *
 * Predict equations:
 *   X_pred[i] = f(X[i], u, dt)
 *   x̂⁻ = Σ Wm[i] · X_pred[i]
 *   P⁻  = Σ Wc[i] · (X_pred[i]−x̂⁻)(X_pred[i]−x̂⁻)ᵀ + Q
 *
 * Update equations:
 *   Z[i] = h(X_pred[i])
 *   z̄    = Σ Wm[i] · Z[i]
 *   S    = Σ Wc[i] · (Z[i]−z̄)(Z[i]−z̄)ᵀ + R
 *   Pxz  = Σ Wc[i] · (X_pred[i]−x̂⁻)(Z[i]−z̄)ᵀ
 *   K    = Pxz · S⁻¹
 *   x̂    = x̂⁻ + K·(z − z̄)
 *   P    = P⁻ − K·S·Kᵀ
 *
 * Robust Cholesky:
 *   Eigen's llt().matrixL() is tried first.
 *   On failure, a relative perturbation (ε × trace(A)/n) is added
 *   to the diagonal and retried — matches the Python/C implementations.
 *
 * Usage (lane tracking)
 * ---------------------
 *   auto ukf = UKFFilter<4,1,2>::make_lane_filter(dt, sp, sigma_camera);
 *   ukf.initialize(x0, P0);
 *
 *   for each frame:
 *       ukf.predict(u, dt);
 *       if (meas_valid)
 *           ukf.update(z);
 */

#pragma once

#include "FilterBase.hpp"
#include "LaneModel.hpp"

#include <functional>
#include <array>
#include <cmath>
#include <stdexcept>

namespace kf {

// ---------------------------------------------------------------------------
// UKFFilter<N, M, U>
// ---------------------------------------------------------------------------

template <int N, int M, int U>
class UKFFilter : public FilterBase<UKFFilter<N,M,U>, N, M> {
public:
    using Base      = FilterBase<UKFFilter<N,M,U>, N, M>;
    using StateVec  = typename Base::StateVec;
    using StateMat  = typename Base::StateMat;
    using ObsVec    = typename Base::ObsVec;
    using ObsMat    = typename Base::ObsMat;
    using GainMat   = typename Base::GainMat;
    using CtrlVec   = Eigen::Matrix<double, U, 1>;

    static constexpr int NS = 2 * N + 1;  ///< number of sigma points

    // Sigma-point matrices — reused between predict and update
    using SigmaMat  = Eigen::Matrix<double, N, NS>;   // [N × 2N+1]
    using ObsSigMat = Eigen::Matrix<double, M, NS>;   // [M × 2N+1]

    // ------------------------------------------------------------------
    // Nonlinear functions
    // ------------------------------------------------------------------
    std::function<StateVec(const StateVec&, const CtrlVec&, double)> f_func;
    std::function<ObsVec(const StateVec&)>                           h_func;

    const char* filter_name() const { return "UKF"; }

    // ------------------------------------------------------------------
    // Sigma-point parameters (set by factory)
    // ------------------------------------------------------------------
    double alpha  = 1e-3;
    double beta   = 2.0;
    double kappa  = 0.0;
    double lambda = 0.0;

    // Precomputed weights [NS]
    std::array<double, NS> Wm{};
    std::array<double, NS> Wc{};

    // Cache from predict(), consumed by update()
    SigmaMat X_pred = SigmaMat::Zero();
    StateVec x_prior = StateVec::Zero();

    // ------------------------------------------------------------------
    // Factory
    // ------------------------------------------------------------------
    static UKFFilter make_lane_filter(double              dt,
                                      const SigmaProcess& sp,
                                      double              sigma_camera,
                                      double alpha_ = 1e-3,
                                      double beta_  = 2.0,
                                      double kappa_ = 0.0)
    {
        (void)dt;
        UKFFilter ukf;

        ukf.Q = lane::make_Q(sp.C0, sp.C1, sp.C2, sp.C3);
        ukf.R = lane::make_R(sigma_camera);
        ukf.P = lane::make_P0(sp.C0, sp.C1, sp.C2, sp.C3);

        ukf.alpha  = alpha_;
        ukf.beta   = beta_;
        ukf.kappa  = kappa_;
        ukf.lambda = alpha_ * alpha_ * (N + kappa_) - N;
        ukf._compute_weights();

        ukf.f_func = [](const StateVec& x, const CtrlVec& u, double dt_) {
            return lane::f(x, u, dt_);
        };
        ukf.h_func = [](const StateVec& x) {
            return lane::h(x);
        };

        return ukf;
    }

    // ------------------------------------------------------------------
    // CRTP predict implementation
    // ------------------------------------------------------------------
    /**
     * Unscented time update.
     *
     * 1. Cholesky of (n+λ)·P  →  L
     * 2. Generate 2n+1 sigma points
     * 3. Propagate each through f
     * 4. Recover x̂⁻ and P⁻ from weighted sigma points
     *
     * @param u   Control [v, ω]
     * @param dt  Sample period (s)
     * @return    Predicted state x̂⁻
     */
    const StateVec& predict_impl(const CtrlVec& u, double dt)
    {
        // --- Step 1: Cholesky of scaled covariance ---
        const StateMat scaled_P = (N + lambda) * this->P;
        StateMat L = _robust_llt(scaled_P);

        // --- Step 2: Sigma points ---
        // X[:, 0] = x̂
        // X[:, i]   = x̂ + L[:, i-1]       i = 1…n
        // X[:, i+n] = x̂ − L[:, i-1]       i = 1…n
        SigmaMat X;
        X.col(0) = this->x;
        for (int i = 0; i < N; ++i) {
            X.col(i + 1)     = this->x + L.col(i);
            X.col(i + 1 + N) = this->x - L.col(i);
        }

        // --- Step 3: Propagate sigma points through f ---
        for (int i = 0; i < NS; ++i)
            X_pred.col(i) = f_func(X.col(i), u, dt);

        // --- Step 4: Weighted mean ---
        x_prior = StateVec::Zero();
        for (int i = 0; i < NS; ++i)
            x_prior += Wm[i] * X_pred.col(i);

        // --- Step 5: Weighted covariance ---
        StateMat P_prior = this->Q;
        for (int i = 0; i < NS; ++i) {
            const StateVec d = X_pred.col(i) - x_prior;
            P_prior += Wc[i] * d * d.transpose();
        }

        this->x = x_prior;
        this->P = P_prior;
        return this->x;
    }

    // ------------------------------------------------------------------
    // CRTP update implementation
    // ------------------------------------------------------------------
    /**
     * Unscented measurement update.
     * Reuses X_pred and x_prior from the preceding predict() call.
     *
     * @param z  Observation [M]
     * @return   true on success, false if S is singular.
     */
    bool update_impl(const ObsVec& z)
    {
        // --- Propagate sigma points through h ---
        ObsSigMat Z_pred;
        for (int i = 0; i < NS; ++i)
            Z_pred.col(i) = h_func(X_pred.col(i));

        // --- Predicted measurement mean ---
        ObsVec z_bar = ObsVec::Zero();
        for (int i = 0; i < NS; ++i)
            z_bar += Wm[i] * Z_pred.col(i);

        // --- Innovation covariance S ---
        ObsMat S_mat = this->R;
        for (int i = 0; i < NS; ++i) {
            const ObsVec dz = Z_pred.col(i) - z_bar;
            S_mat += Wc[i] * dz * dz.transpose();
        }
        this->S = S_mat;

        // --- Cross-covariance Pxz [N×M] ---
        GainMat Pxz = GainMat::Zero();
        for (int i = 0; i < NS; ++i) {
            const StateVec dx = X_pred.col(i) - x_prior;
            const ObsVec   dz = Z_pred.col(i) - z_bar;
            Pxz += Wc[i] * dx * dz.transpose();
        }

        // --- Kalman gain K = Pxz · S⁻¹ ---
        // For M==1 the innovation covariance S is a scalar — avoid LDLT
        // (which triggers spurious GCC -Warray-bounds via SSE code paths)
        // and divide directly.  For M>1 use LDLT (symmetric positive-definite).
        this->innovation = z - z_bar;
        if constexpr (M == 1) {
            const double S_val = S_mat(0, 0);
            if (S_val < 1e-12) return false;
            this->K   = Pxz * (1.0 / S_val);
            this->NIS = (this->innovation(0) * this->innovation(0)) / S_val;
        } else {
            Eigen::LDLT<ObsMat> ldlt(S_mat);
            if (ldlt.info() != Eigen::Success) return false;
            this->K   = ldlt.solve(Pxz.transpose()).transpose();
            this->NIS = (this->innovation.transpose()
                         * ldlt.solve(this->innovation)).value();
        }

        // --- State update ---
        this->x = this->x + this->K * this->innovation;

        // --- Covariance downdate: P = P⁻ − K·S·Kᵀ ---
        this->P = this->P - this->K * S_mat * this->K.transpose();

        // Enforce symmetry
        this->P = (this->P + this->P.transpose()) * 0.5;

        return true;
    }

private:
    // ------------------------------------------------------------------
    // Weight initialisation
    // ------------------------------------------------------------------
    void _compute_weights()
    {
        const double w_common = 0.5 / (N + lambda);
        Wm[0] = lambda / (N + lambda);
        Wc[0] = lambda / (N + lambda) + (1.0 - alpha * alpha + beta);
        for (int i = 1; i < NS; ++i) { Wm[i] = w_common; Wc[i] = w_common; }
    }

    // ------------------------------------------------------------------
    // Robust Cholesky: try LLT, on failure add relative diagonal eps
    // ------------------------------------------------------------------
    static StateMat _robust_llt(const StateMat& A)
    {
        Eigen::LLT<StateMat> llt(A);
        if (llt.info() == Eigen::Success)
            return llt.matrixL();

        // Relative perturbation: eps = 1e-9 × trace(A)/N
        const double eps = 1e-9 * A.trace() / N;
        for (double scale : {1.0, 1e2, 1e4, 1e6}) {
            StateMat reg = A + (eps * scale) * StateMat::Identity();
            Eigen::LLT<StateMat> llt2(reg);
            if (llt2.info() == Eigen::Success)
                return llt2.matrixL();
        }
        // Last resort: return identity scaled by a safe value
        return StateMat::Identity() * std::sqrt(A.trace() / N + 1e-8);
    }
};

// Convenience alias
using LaneUKF = UKFFilter<lane::N, lane::M, lane::U>;

} // namespace kf
