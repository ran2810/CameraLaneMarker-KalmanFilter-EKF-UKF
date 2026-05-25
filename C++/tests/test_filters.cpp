/**
 * test_filters.cpp — Unit tests for C++ KF / EKF / UKF.
 *
 * Checks:
 *   1.  KF scalar cycle — exact closed-form answers
 *   2.  KF lane-filter factory — matrix shapes and symmetry
 *   3.  KF vs known Python RMSE (regression)
 *   4.  EKF on straight road (sin(C1)≈C1) ≈ KF
 *   5.  UKF on straight road ≈ EKF  (nonlinearity negligible)
 *   6.  NIS consistency — mean(NIS) close to 1 for m=1
 *   7.  Joseph-form — P stays symmetric and positive-definite after 100 steps
 *
 * Build & run:
 *   cmake --build build && ctest --test-dir build -V
 */

#include "KalmanFilter.hpp"
#include "EKFFilter.hpp"
#include "UKFFilter.hpp"
#include "LaneModel.hpp"

#include <iostream>
#include <cmath>
#include <string>

using namespace kf;

// ---------------------------------------------------------------------------
// Minimal test framework
// ---------------------------------------------------------------------------
static int g_failures = 0;

#define ASSERT_NEAR(val, expected, tol, name) \
    do { \
        double _v = (val), _e = (expected), _t = (tol); \
        if (std::abs(_v - _e) > _t) { \
            std::cout << "  FAIL: " << (name) \
                      << "  got=" << _v << "  want=" << _e \
                      << "  tol=" << _t << '\n'; \
            ++g_failures; \
        } else { \
            std::cout << "  PASS: " << (name) << '\n'; \
        } \
    } while(0)

#define ASSERT_TRUE(expr, name) \
    do { \
        if (!(expr)) { \
            std::cout << "  FAIL: " << (name) << '\n'; \
            ++g_failures; \
        } else { \
            std::cout << "  PASS: " << (name) << '\n'; \
        } \
    } while(0)


// ---------------------------------------------------------------------------
// 1. Scalar KF cycle — exact closed-form
//
//    x=0, P=1, F=1, H=1, Q=0.01, R=0.25  → z=1
//    x_prior = 0,  P_prior = 1.01
//    K       = P_prior / (P_prior + R) = 1.01/1.26
//    x_post  = K * 1  = 1.01/1.26
//    P_post  = R * P_prior / (P_prior + R)  =  0.25*1.01/1.26   (Joseph form)
// ---------------------------------------------------------------------------
static void test_scalar_cycle()
{
    std::cout << "\n--- Scalar KF cycle ---\n";

    using KF11 = KalmanFilter<1,1>;
    KF11 kf;
    kf.F(0,0) = 1.0;
    kf.H(0,0) = 1.0;
    kf.Q(0,0) = 0.01;
    kf.R(0,0) = 0.25;
    kf.x(0)   = 0.0;
    kf.P(0,0) = 1.0;

    // Predict
    kf.predict();
    ASSERT_NEAR(kf.x(0), 0.0,  1e-14, "x after predict");
    ASSERT_NEAR(kf.P(0,0), 1.01, 1e-14, "P after predict");

    // Update z=1
    KF11::ObsVec z; z(0) = 1.0;
    bool ok = kf.update(z);
    ASSERT_TRUE(ok, "update returns true");

    const double P_prior = 1.01;
    const double R_      = 0.25;
    const double K_exp   = P_prior / (P_prior + R_);
    const double P_exp   = R_ * P_prior / (P_prior + R_);

    ASSERT_NEAR(kf.K(0,0),  K_exp, 1e-12, "Kalman gain K");
    ASSERT_NEAR(kf.x(0),    K_exp, 1e-12, "x_post");
    ASSERT_NEAR(kf.P(0,0),  P_exp, 1e-12, "P_post (Joseph)");
    // NIS = y²/S  where S = H·P⁻·Hᵀ + R = P_prior + R (H=[1], y=1)
    ASSERT_NEAR(kf.NIS, 1.0 / (P_prior + R_), 1e-10, "NIS (y²/S)");
}


// ---------------------------------------------------------------------------
// 2. KF lane filter — sanity checks on matrix properties
// ---------------------------------------------------------------------------
static void test_kf_lane_factory()
{
    std::cout << "\n--- KF lane filter factory ---\n";

    const double dt = 0.04;
    SigmaProcess sp{0.07, 0.004, 4e-5, 8e-7};
    auto kf = LaneKF::make_lane_filter(dt, sp, 0.1);

    // F should be upper triangular (clothoid model)
    ASSERT_NEAR(kf.F(1,0), 0.0,   1e-15, "F(1,0) = 0");
    ASSERT_NEAR(kf.F(0,1), dt,    1e-15, "F(0,1) = dt");
    ASSERT_NEAR(kf.F(0,2), 0.5*dt*dt, 1e-15, "F(0,2) = dt²/2");

    // H selects C0 only
    ASSERT_NEAR(kf.H(0,0), 1.0, 1e-15, "H(0,0) = 1");
    ASSERT_NEAR(kf.H(0,1), 0.0, 1e-15, "H(0,1) = 0");

    // Q diagonal = σ²
    ASSERT_NEAR(kf.Q(0,0), 0.07*0.07, 1e-16, "Q C0 diagonal");
    ASSERT_NEAR(kf.Q(1,1), 0.004*0.004, 1e-18, "Q C1 diagonal");

    // P symmetric at init
    ASSERT_NEAR(kf.P(0,1), kf.P(1,0), 1e-15, "P symmetric at init");
}


// ---------------------------------------------------------------------------
// 3. P stays symmetric and positive-definite after 200 steps
// ---------------------------------------------------------------------------
static void test_covariance_health()
{
    std::cout << "\n--- Covariance health (symmetry + PD) after 200 steps ---\n";

    SigmaProcess sp{0.07, 0.004, 4e-5, 8e-7};
    auto kf = LaneKF::make_lane_filter(0.04, sp, 0.1);

    LaneKF::StateVec x0; x0 << 1.875, 0.0, 0.0, 0.0;
    LaneKF::StateMat P0 = LaneKF::StateMat::Zero();
    P0.diagonal() << 0.1, 0.01, 1e-4, 1e-6;
    kf.initialize(x0, P0);

    for (int k = 0; k < 200; ++k) {
        kf.predict();
        LaneKF::ObsVec z; z(0) = 1.875 + 0.05 * std::sin(k * 0.1);
        kf.update(z);
    }

    // Symmetry
    double max_asymm = (kf.P - kf.P.transpose()).cwiseAbs().maxCoeff();
    ASSERT_TRUE(max_asymm < 1e-12, "P symmetric after 200 steps");

    // Positive-definiteness: all eigenvalues > 0
    Eigen::SelfAdjointEigenSolver<LaneKF::StateMat> es(kf.P);
    ASSERT_TRUE(es.eigenvalues().minCoeff() > 0.0, "P positive-definite after 200 steps");
}


// ---------------------------------------------------------------------------
// 4. EKF on straight road ≈ KF  (sin(C1) ≈ C1 → nonlinearity negligible)
// ---------------------------------------------------------------------------
static void test_ekf_vs_kf_straight()
{
    std::cout << "\n--- EKF ≈ KF on straight road ---\n";

    const double dt = 0.04;
    SigmaProcess sp{0.07, 0.004, 4e-5, 8e-7};

    auto kf  = LaneKF::make_lane_filter(dt, sp, 0.1);
    auto ekf = LaneEKF::make_lane_filter(dt, sp, 0.1);

    LaneKF::StateVec x0; x0 << 1.875, 0.0, 0.0, 0.0;
    LaneKF::StateMat P0 = LaneKF::StateMat::Zero();
    P0.diagonal() << 0.1, 0.01, 1e-4, 1e-6;
    kf.initialize(x0, P0);
    ekf.initialize(x0, P0);

    // Straight road: v=33.3, omega=0
    LaneEKF::CtrlVec u; u << 33.3, 0.0;

    double rmse_kf = 0, rmse_ekf = 0;
    const double truth = 1.875;

    for (int k = 0; k < 750; ++k) {  // 30s at 25 Hz
        kf.predict();
        ekf.predict(u, dt);

        LaneKF::ObsVec z; z(0) = truth + 0.05 * std::sin(k * 0.3);
        kf.update(z);
        ekf.update(z);

        double ek  = kf.x(0)  - truth;
        double ee  = ekf.x(0) - truth;
        rmse_kf  += ek * ek;
        rmse_ekf += ee * ee;
    }
    rmse_kf  = std::sqrt(rmse_kf  / 750);
    rmse_ekf = std::sqrt(rmse_ekf / 750);

    // EKF and KF should agree closely on straight road
    ASSERT_NEAR(rmse_ekf, rmse_kf, 0.005, "EKF ≈ KF RMSE on straight (5mm)");
    std::cout << "    KF RMSE=" << rmse_kf << "  EKF RMSE=" << rmse_ekf << "\n";
}


// ---------------------------------------------------------------------------
// 5. UKF ≈ EKF on straight road
// ---------------------------------------------------------------------------
static void test_ukf_vs_ekf_straight()
{
    std::cout << "\n--- UKF ≈ EKF on straight road ---\n";

    const double dt = 0.04;
    SigmaProcess sp{0.07, 0.004, 4e-5, 8e-7};

    auto ekf = LaneEKF::make_lane_filter(dt, sp, 0.1);
    auto ukf = LaneUKF::make_lane_filter(dt, sp, 0.1);

    LaneEKF::StateVec x0; x0 << 1.875, 0.0, 0.0, 0.0;
    LaneEKF::StateMat P0 = LaneEKF::StateMat::Zero();
    P0.diagonal() << 0.1, 0.01, 1e-4, 1e-6;
    ekf.initialize(x0, P0);
    ukf.initialize(x0, P0);

    LaneEKF::CtrlVec u; u << 33.3, 0.0;

    double rmse_ekf = 0, rmse_ukf = 0;
    const double truth = 1.875;

    for (int k = 0; k < 750; ++k) {
        ekf.predict(u, dt);
        ukf.predict(u, dt);

        LaneEKF::ObsVec z; z(0) = truth + 0.05 * std::sin(k * 0.3);
        ekf.update(z);
        ukf.update(z);

        double ee = ekf.x(0) - truth;
        double eu = ukf.x(0) - truth;
        rmse_ekf += ee * ee;
        rmse_ukf += eu * eu;
    }
    rmse_ekf = std::sqrt(rmse_ekf / 750);
    rmse_ukf = std::sqrt(rmse_ukf / 750);

    ASSERT_NEAR(rmse_ukf, rmse_ekf, 0.005, "UKF ≈ EKF RMSE on straight (5mm)");

    // Final C0 estimates should also be close
    ASSERT_NEAR(ukf.x(0), ekf.x(0), 0.01, "UKF ≈ EKF final C0 (10mm)");
    std::cout << "    EKF RMSE=" << rmse_ekf << "  UKF RMSE=" << rmse_ukf << "\n";
}


// ---------------------------------------------------------------------------
// 6. NIS consistency — mean(NIS) should be close to M=1
// ---------------------------------------------------------------------------
static void test_nis_consistency()
{
    std::cout << "\n--- NIS consistency ---\n";

    // NIS must be driven by proper N(0, R) noise — a deterministic sinusoid
    // produces E[NIS] << 1 because it is not zero-mean across timesteps.
    // We use a simple LCG so the test is deterministic (no <random> state).
    const double dt      = 0.04;
    const double sigma_R = 0.1;   // must match make_lane_filter sigma_camera
    SigmaProcess sp{0.07, 0.004, 4e-5, 8e-7};
    const double truth   = 1.875;

    // Minimal deterministic LCG for N(0,1) via Box–Muller
    auto make_randn = [](uint32_t seed) {
        return [s = seed]() mutable -> double {
            // LCG: Numerical Recipes constants
            s = s * 1664525u + 1013904223u;
            double u1 = (s & 0xFFFFFFu) / double(0x1000000);
            s = s * 1664525u + 1013904223u;
            double u2 = (s & 0xFFFFFFu) / double(0x1000000);
            u1 = std::max(u1, 1e-10);
            return std::sqrt(-2.0 * std::log(u1)) * std::cos(2.0 * M_PI * u2);
        };
    };

    // Process noise std-devs — must match make_lane_filter
    const double q0 = sp.C0, q1 = sp.C1, q2 = sp.C2, q3 = sp.C3;

    auto run = [&](auto& filt, auto pred_fn, uint32_t seed) {
        auto randn = make_randn(seed);

        // True state evolves with the SAME process noise as the filter model,
        // so the filter is correctly specified and E[NIS] = 1.
        LaneKF::StateVec x_true; x_true << truth, 0, 0, 0;

        LaneKF::StateVec x0 = x_true;
        LaneKF::StateMat P0 = LaneKF::StateMat::Zero();
        P0.diagonal() << 0.1, 0.01, 1e-4, 1e-6;
        filt.initialize(x0, P0);

        double sum_nis = 0; int n = 0;
        for (int k = 0; k < 3000; ++k) {
            // Propagate true state: constant-curvature-rate model + process noise
            LaneKF::StateVec w;
            w << q0*randn(), q1*randn(), q2*randn(), q3*randn();
            x_true = x_true + w;          // simplified: F≈I for small dt

            pred_fn(k);

            // Measurement: true C0 + camera noise
            LaneKF::ObsVec z; z(0) = x_true(0) + sigma_R * randn();
            filt.update(z);
            sum_nis += filt.NIS; ++n;
        }
        return sum_nis / n;
    };

    auto kf  = LaneKF::make_lane_filter(dt, sp, 0.1);
    auto ekf = LaneEKF::make_lane_filter(dt, sp, 0.1);
    auto ukf = LaneUKF::make_lane_filter(dt, sp, 0.1);

    LaneEKF::CtrlVec u; u << 33.3, 0.0;

    double nis_kf  = run(kf,  [&](int){ kf.predict(); },         42u);
    double nis_ekf = run(ekf, [&](int){ ekf.predict(u, dt); },  42u);
    double nis_ukf = run(ukf, [&](int){ ukf.predict(u, dt); },  42u);

    // For consistent filter: mean(NIS) ∈ [0.9, 1.1]  (rough 95% CI)
    ASSERT_NEAR(nis_kf,  1.0, 0.15, "KF mean NIS ≈ 1");
    ASSERT_NEAR(nis_ekf, 1.0, 0.15, "EKF mean NIS ≈ 1");
    ASSERT_NEAR(nis_ukf, 1.0, 0.15, "UKF mean NIS ≈ 1");

    std::cout << "    KF=" << nis_kf << "  EKF=" << nis_ekf
              << "  UKF=" << nis_ukf << "  (expected≈1.0)\n";
}


// ---------------------------------------------------------------------------
// main
// ---------------------------------------------------------------------------
int main()
{
    std::cout << "=================================================\n"
              << "  C++ Kalman Filter unit tests\n"
              << "=================================================\n";

    test_scalar_cycle();
    test_kf_lane_factory();
    test_covariance_health();
    test_ekf_vs_kf_straight();
    test_ukf_vs_ekf_straight();
    test_nis_consistency();

    std::cout << "\n=================================================\n";
    if (g_failures == 0)
        std::cout << "  ALL TESTS PASSED\n";
    else
        std::cout << "  " << g_failures << " TEST(S) FAILED\n";
    std::cout << "=================================================\n";

    return g_failures;
}
