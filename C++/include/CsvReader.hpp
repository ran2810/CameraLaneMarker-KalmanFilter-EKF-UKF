/**
 * CsvReader.hpp — Lightweight CSV parser for the lane dataset.
 *
 * Reads the CSV produced by generate_lane_data.py (or ds.save_csv()).
 * Uses only the C++ standard library — no third-party deps.
 *
 * Expected header:
 *   t,speed,yaw_rate,C0_L_true,C0_R_true,C1_true,C2_true,C3_true,meas_L,meas_R
 *
 * meas_L/meas_R fields may be "nan" — these are stored as NaN (std::nan("")).
 */

#pragma once

#include <string>
#include <vector>
#include <fstream>
#include <sstream>
#include <stdexcept>
#include <cmath>

namespace kf {

// ---------------------------------------------------------------------------
// Frame — one row of the lane dataset
// ---------------------------------------------------------------------------
struct Frame {
    double t          = 0.0;
    double speed      = 0.0;
    double yaw_rate   = 0.0;
    double C0_L_true  = 0.0;
    double C0_R_true  = 0.0;
    double C1_true    = 0.0;
    double C2_true    = 0.0;
    double C3_true    = 0.0;
    double meas_L     = std::numeric_limits<double>::quiet_NaN();
    double meas_R     = std::numeric_limits<double>::quiet_NaN();

    bool meas_L_valid() const { return !std::isnan(meas_L); }
    bool meas_R_valid() const { return !std::isnan(meas_R); }
};

// ---------------------------------------------------------------------------
// CsvReader
// ---------------------------------------------------------------------------
class CsvReader {
public:
    /**
     * Load the entire CSV into memory.
     * @throws std::runtime_error on file open failure.
     */
    static std::vector<Frame> load(const std::string& path)
    {
        std::ifstream f(path);
        if (!f.is_open())
            throw std::runtime_error("CsvReader: cannot open '" + path + "'");

        std::string line;
        std::getline(f, line);   // skip header

        std::vector<Frame> frames;
        frames.reserve(4096);

        while (std::getline(f, line)) {
            if (line.empty()) continue;
            frames.push_back(_parse_row(line));
        }
        return frames;
    }

private:
    static double _parse_field(const std::string& s)
    {
        if (s == "nan" || s == "NaN" || s == "NAN")
            return std::numeric_limits<double>::quiet_NaN();
        return std::stod(s);
    }

    static Frame _parse_row(const std::string& line)
    {
        std::istringstream ss(line);
        std::string tok;
        std::vector<double> vals;
        vals.reserve(10);

        while (std::getline(ss, tok, ','))
            vals.push_back(_parse_field(tok));

        if (vals.size() < 10)
            throw std::runtime_error("CsvReader: malformed row: " + line);

        Frame fr;
        fr.t         = vals[0];
        fr.speed     = vals[1];
        fr.yaw_rate  = vals[2];
        fr.C0_L_true = vals[3];
        fr.C0_R_true = vals[4];
        fr.C1_true   = vals[5];
        fr.C2_true   = vals[6];
        fr.C3_true   = vals[7];
        fr.meas_L    = vals[8];
        fr.meas_R    = vals[9];
        return fr;
    }
};

// ---------------------------------------------------------------------------
// CsvWriter — write per-frame filter estimates
// ---------------------------------------------------------------------------
class CsvWriter {
public:
    explicit CsvWriter(const std::string& path)
        : f_(path)
    {
        if (!f_.is_open())
            throw std::runtime_error("CsvWriter: cannot open '" + path + "'");

        f_ << "t,"
           << "C0_L_true,meas_L,kf_C0_L,ekf_C0_L,ukf_C0_L,"
           << "kf_NIS_L,ekf_NIS_L,ukf_NIS_L,"
           << "C0_R_true,meas_R,kf_C0_R,ekf_C0_R,ukf_C0_R,"
           << "kf_NIS_R,ekf_NIS_R,ukf_NIS_R\n";
        f_ << std::fixed;
    }

    void write_row(double t,
                   double C0_L_true, double meas_L,
                   double kf_C0_L,  double ekf_C0_L, double ukf_C0_L,
                   double NIS_kf_L, double NIS_ekf_L, double NIS_ukf_L,
                   double C0_R_true, double meas_R,
                   double kf_C0_R,  double ekf_C0_R, double ukf_C0_R,
                   double NIS_kf_R, double NIS_ekf_R, double NIS_ukf_R)
    {
        auto fmt = [](double v) -> std::string {
            if (std::isnan(v)) return "nan";
            std::ostringstream ss;
            ss << std::fixed;
            ss.precision(6);
            ss << v;
            return ss.str();
        };

        f_ << fmt(t)          << ','
           << fmt(C0_L_true)  << ',' << fmt(meas_L)
           << ',' << fmt(kf_C0_L)  << ',' << fmt(ekf_C0_L)  << ',' << fmt(ukf_C0_L)
           << ',' << fmt(NIS_kf_L) << ',' << fmt(NIS_ekf_L) << ',' << fmt(NIS_ukf_L)
           << ',' << fmt(C0_R_true) << ',' << fmt(meas_R)
           << ',' << fmt(kf_C0_R)  << ',' << fmt(ekf_C0_R)  << ',' << fmt(ukf_C0_R)
           << ',' << fmt(NIS_kf_R) << ',' << fmt(NIS_ekf_R) << ',' << fmt(NIS_ukf_R)
           << '\n';
    }

private:
    std::ofstream f_;
};

} // namespace kf
