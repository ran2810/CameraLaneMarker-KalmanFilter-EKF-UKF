/**
 * compare_filters.cpp — KF / EKF / UKF comparison on the highway dataset.
 *
 * Usage:
 *   ./compare_filters <input.csv> [output.csv]
 *
 * Mirrors the output of c/examples/compare_filters.c so the two
 * implementations can be compared side-by-side.
 */

#include "KalmanFilter.hpp"
#include "EKFFilter.hpp"
#include "UKFFilter.hpp"
#include "CsvReader.hpp"
#include "Metrics.hpp"

#include <iostream>
#include <string>
#include <vector>
#include <cmath>
#include <memory>
#include <stdexcept>

using namespace kf;

// ---------------------------------------------------------------------------
// Filter parameters — identical to C and Python
// ---------------------------------------------------------------------------
static constexpr double DT           = 0.04;
static constexpr double SIGMA_CAMERA = 0.1;

static const SigmaProcess SP { 0.07, 0.004, 4e-5, 8e-7 };

// Initial covariance diagonal
static const LaneKF::StateMat make_P0()
{
    LaneKF::StateMat P = LaneKF::StateMat::Zero();
    P.diagonal() << 0.1, 0.01, 1e-4, 1e-6;
    return P;
}

// UKF tuning
static constexpr double UKF_ALPHA = 1e-3;
static constexpr double UKF_BETA  = 2.0;
static constexpr double UKF_KAPPA = 0.0;

// Scenario phase boundaries (seconds)
static const double PHASE_T[] = { 0, 30, 50, 70, 90, 120 };
static const char*  PHASE_N[] = {
    "Ph1 A9 straight",
    "Ph2 Curve      ",
    "Ph3 Transition ",
    "Ph4 A8 straight",
    "Ph5 S-curve    ",
};


// ---------------------------------------------------------------------------
// Helper: build + initialise a filter pair (L / R)
// ---------------------------------------------------------------------------
template <typename FilterType>
static auto build_filter_pair(const std::vector<Frame>& frames,
                               FilterType filter_template)
    -> std::pair<FilterType, FilterType>
{
    const auto P0 = make_P0();

    LaneKF::StateVec x0_L, x0_R;
    x0_L << frames[0].C0_L_true, 0, 0, 0;
    x0_R << frames[0].C0_R_true, 0, 0, 0;

    FilterType fL = filter_template;
    FilterType fR = filter_template;
    fL.initialize(x0_L, P0);
    fR.initialize(x0_R, P0);
    return {fL, fR};
}


// ---------------------------------------------------------------------------
// main
// ---------------------------------------------------------------------------
int main(int argc, char* argv[])
{
    if (argc < 2) {
        std::cerr << "Usage: " << argv[0] << " <input.csv> [output.csv]\n";
        return 1;
    }

    // ---------------------------------------------------------------
    // Load data
    // ---------------------------------------------------------------
    std::vector<Frame> frames;
    try {
        frames = CsvReader::load(argv[1]);
    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << '\n';
        return 1;
    }
    std::cout << "Loaded " << frames.size() << " frames from " << argv[1] << '\n';

    // ---------------------------------------------------------------
    // Build filters
    // ---------------------------------------------------------------
    auto kf_tmpl  = LaneKF::make_lane_filter(DT, SP, SIGMA_CAMERA);
    auto ekf_tmpl = LaneEKF::make_lane_filter(DT, SP, SIGMA_CAMERA);
    auto ukf_tmpl = LaneUKF::make_lane_filter(DT, SP, SIGMA_CAMERA,
                                               UKF_ALPHA, UKF_BETA, UKF_KAPPA);

    auto [kf_L,  kf_R]  = build_filter_pair(frames, kf_tmpl);
    auto [ekf_L, ekf_R] = build_filter_pair(frames, ekf_tmpl);
    auto [ukf_L, ukf_R] = build_filter_pair(frames, ukf_tmpl);

    // ---------------------------------------------------------------
    // Optional output CSV
    // ---------------------------------------------------------------
    std::unique_ptr<CsvWriter> writer;
    if (argc >= 3) {
        try {
            writer = std::make_unique<CsvWriter>(argv[2]);
        } catch (const std::exception& e) {
            std::cerr << "Warning: " << e.what() << " — continuing without CSV output\n";
        }
    }

    // ---------------------------------------------------------------
    // Metrics: overall + per-phase
    // ---------------------------------------------------------------
    RunningMetrics overall_kf_L,  overall_ekf_L,  overall_ukf_L;
    RunningMetrics overall_kf_R,  overall_ekf_R,  overall_ukf_R;

    static constexpr int N_PHASES = 5;
    std::vector<PhaseMetrics> phase_metrics(N_PHASES);
    for (int i = 0; i < N_PHASES; ++i)
        phase_metrics[i].name = PHASE_N[i];

    // Control vector placeholder
    LaneEKF::CtrlVec u;

    // ---------------------------------------------------------------
    // Main filter loop
    // ---------------------------------------------------------------
    for (size_t k = 0; k < frames.size(); ++k) {
        const Frame& fr = frames[k];

        u << fr.speed, fr.yaw_rate;

        // --- Predict ---
        kf_L.predict();            kf_R.predict();
        ekf_L.predict(u, DT);     ekf_R.predict(u, DT);
        bool ukf_L_ok = ukf_L.predict(u, DT).allFinite();
        bool ukf_R_ok = ukf_R.predict(u, DT).allFinite();

        // --- Update ---
        double NIS_kf_L = std::numeric_limits<double>::quiet_NaN();
        double NIS_ekf_L = NIS_kf_L, NIS_ukf_L = NIS_kf_L;
        double NIS_kf_R = NIS_kf_L;
        double NIS_ekf_R = NIS_kf_L, NIS_ukf_R = NIS_kf_L;

        LaneKF::ObsVec z;
        if (fr.meas_L_valid()) {
            z(0) = fr.meas_L;
            if (kf_L.update(z))                   NIS_kf_L  = kf_L.NIS;
            if (ekf_L.update(z))                  NIS_ekf_L = ekf_L.NIS;
            if (ukf_L_ok && ukf_L.update(z))      NIS_ukf_L = ukf_L.NIS;
        }
        if (fr.meas_R_valid()) {
            z(0) = fr.meas_R;
            if (kf_R.update(z))                   NIS_kf_R  = kf_R.NIS;
            if (ekf_R.update(z))                  NIS_ekf_R = ekf_R.NIS;
            if (ukf_R_ok && ukf_R.update(z))      NIS_ukf_R = ukf_R.NIS;
        }

        // --- Accumulate overall metrics ---
        overall_kf_L.update(kf_L.x(0),  fr.C0_L_true, NIS_kf_L);
        overall_ekf_L.update(ekf_L.x(0), fr.C0_L_true, NIS_ekf_L);
        overall_ukf_L.update(ukf_L.x(0), fr.C0_L_true, NIS_ukf_L);

        overall_kf_R.update(kf_R.x(0),  fr.C0_R_true, NIS_kf_R);
        overall_ekf_R.update(ekf_R.x(0), fr.C0_R_true, NIS_ekf_R);
        overall_ukf_R.update(ukf_R.x(0), fr.C0_R_true, NIS_ukf_R);

        // --- Accumulate per-phase metrics ---
        for (int ph = 0; ph < N_PHASES; ++ph) {
            if (fr.t >= PHASE_T[ph] && fr.t < PHASE_T[ph+1]) {
                phase_metrics[ph].kf.update(kf_L.x(0),  fr.C0_L_true, NIS_kf_L);
                phase_metrics[ph].ekf.update(ekf_L.x(0), fr.C0_L_true, NIS_ekf_L);
                phase_metrics[ph].ukf.update(ukf_L.x(0), fr.C0_L_true, NIS_ukf_L);
                break;
            }
        }

        // --- Console heartbeat (every 1 s) ---
        if (k % 25 == 0) {
            std::cout << std::fixed << std::setprecision(2)
                      << "t=" << fr.t
                      << "s  C0_L true=" << fr.C0_L_true
                      << "  kf=" << kf_L.x(0)
                      << "  ekf=" << ekf_L.x(0)
                      << "  ukf=" << ukf_L.x(0)
                      << '\n';
        }

        // --- Write CSV row ---
        if (writer) {
            writer->write_row(
                fr.t,
                fr.C0_L_true, fr.meas_L,
                kf_L.x(0), ekf_L.x(0), ukf_L.x(0),
                NIS_kf_L, NIS_ekf_L, NIS_ukf_L,
                fr.C0_R_true, fr.meas_R,
                kf_R.x(0), ekf_R.x(0), ukf_R.x(0),
                NIS_kf_R, NIS_ekf_R, NIS_ukf_R
            );
        }
    }

    // ---------------------------------------------------------------
    // Summary output
    // ---------------------------------------------------------------
    print_summary("Overall — Left Marker (C0_L)",
                  overall_kf_L, overall_ekf_L, overall_ukf_L);
    print_summary("Overall — Right Marker (C0_R)",
                  overall_kf_R, overall_ekf_R, overall_ukf_R);
    print_phase_table(phase_metrics);

    if (argc >= 3)
        std::cout << "\nOutput → " << argv[2] << '\n';

    return 0;
}
