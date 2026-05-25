/**
 * Metrics.hpp — Filter performance metrics.
 *
 * RunningMetrics  — accumulates RMSE / MAE / NIS online (one pass)
 * print_summary() — prints the comparison table
 */

#pragma once

#include <cmath>
#include <string>
#include <vector>
#include <iostream>
#include <iomanip>
#include <numeric>

namespace kf {

// ---------------------------------------------------------------------------
// RunningMetrics — accumulated online
// ---------------------------------------------------------------------------
struct RunningMetrics {
    double sum_sq_err  = 0.0;
    double sum_abs_err = 0.0;
    double sum_nis     = 0.0;
    int    n_err       = 0;
    int    n_nis       = 0;

    void update(double estimate, double truth, double NIS)
    {
        if (!std::isnan(estimate) && !std::isnan(truth)) {
            double e = estimate - truth;
            sum_sq_err  += e * e;
            sum_abs_err += std::abs(e);
            ++n_err;
        }
        if (!std::isnan(NIS)) {
            sum_nis += NIS;
            ++n_nis;
        }
    }

    double rmse()     const { return n_err > 0 ? std::sqrt(sum_sq_err / n_err)  : 0.0; }
    double mae()      const { return n_err > 0 ? sum_abs_err / n_err            : 0.0; }
    double mean_nis() const { return n_nis > 0 ? sum_nis / n_nis               : 0.0; }
};

// ---------------------------------------------------------------------------
// Summary printer (mirrors Python print_metrics_table)
// ---------------------------------------------------------------------------
inline void print_summary(const std::string&    side,
                           const RunningMetrics& kf_m,
                           const RunningMetrics& ekf_m,
                           const RunningMetrics& ukf_m)
{
    const int W = 60;
    std::cout << '\n';
    for (int i=0;i<W;++i) std::cout << '=';
    std::cout << "\n  " << side << '\n';
    for (int i=0;i<W;++i) std::cout << '=';
    std::cout << '\n';

    auto row = [&](const std::string& name, const RunningMetrics& m) {
        std::cout << std::fixed << std::setprecision(4)
                  << "  " << std::left << std::setw(6) << name
                  << "  RMSE=" << std::setw(7) << m.rmse()
                  << " m   MAE=" << std::setw(7) << m.mae()
                  << " m   mean_NIS=" << std::setw(6) << m.mean_nis()
                  << '\n';
    };

    row("KF",  kf_m);
    row("EKF", ekf_m);
    row("UKF", ukf_m);
}

// ---------------------------------------------------------------------------
// Per-phase table
// ---------------------------------------------------------------------------
struct PhaseMetrics {
    std::string name;
    RunningMetrics kf, ekf, ukf;
};

inline void print_phase_table(const std::vector<PhaseMetrics>& phases)
{
    const int W = 62;
    std::cout << '\n';
    for (int i=0;i<W;++i) std::cout << '=';
    std::cout << "\n  Per-phase RMSE — Left Marker (m)\n"
              << "  " << std::left << std::setw(20) << "Phase"
              << std::right
              << std::setw(10) << "KF"
              << std::setw(10) << "EKF"
              << std::setw(10) << "UKF" << '\n';
    for (int i=0;i<W;++i) std::cout << '=';
    std::cout << '\n';

    for (const auto& ph : phases) {
        std::cout << std::fixed << std::setprecision(4)
                  << "  " << std::left << std::setw(20) << ph.name
                  << std::right
                  << std::setw(10) << ph.kf.rmse()
                  << std::setw(10) << ph.ekf.rmse()
                  << std::setw(10) << ph.ukf.rmse() << '\n';
    }
}

} // namespace kf
