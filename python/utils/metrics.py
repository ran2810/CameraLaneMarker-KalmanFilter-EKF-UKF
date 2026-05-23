"""
metrics.py — Filter performance metrics

RMSE:   Root Mean Squared Error  — absolute accuracy
MAE:    Mean Absolute Error      — robust to outliers
NIS:    Normalised Innovation Squared — filter consistency check
NEES:   Normalised Estimation Error Squared — consistency (needs ground truth)
"""

import numpy as np
from typing import Optional


def rmse(estimated: np.ndarray, truth: np.ndarray) -> float:
    """Root Mean Squared Error over valid (non-NaN) samples."""
    valid = ~(np.isnan(estimated) | np.isnan(truth))
    return float(np.sqrt(np.mean((estimated[valid] - truth[valid]) ** 2)))


def mae(estimated: np.ndarray, truth: np.ndarray) -> float:
    """Mean Absolute Error over valid samples."""
    valid = ~(np.isnan(estimated) | np.isnan(truth))
    return float(np.mean(np.abs(estimated[valid] - truth[valid])))


def nis_statistics(nis_values: np.ndarray, obs_dim: int) -> dict:
    """
    Summarise NIS sequence and check chi² consistency.

    A filter is statistically consistent if:
        mean(NIS) ≈ obs_dim   (within the 95% chi² confidence interval)

    Args:
        nis_values:  Array of per-step NIS scalars [T].
        obs_dim:     Observation dimension m.

    Returns:
        dict with mean_nis, expected, ratio, consistent flag, bounds.
    """
    from scipy.stats import chi2

    T = len(nis_values)
    mean_nis  = float(np.mean(nis_values))
    expected  = float(obs_dim)

    # 95% confidence interval for the sample mean of chi²(m) variables
    lo = chi2.ppf(0.025, df=obs_dim * T) / T
    hi = chi2.ppf(0.975, df=obs_dim * T) / T

    return {
        "mean_NIS":    mean_nis,
        "expected":    expected,
        "ratio":       mean_nis / expected,
        "consistent":  lo <= mean_nis <= hi,
        "ci_lo":       lo,
        "ci_hi":       hi,
    }


def print_metrics_table(results: dict) -> None:
    """
    Pretty-print a comparison table.

    Args:
        results: {filter_name: {"rmse_L": ..., "rmse_R": ..., "mae_L": ..., "mae_R": ...}}
    """
    header = f"{'Filter':<8} | {'RMSE_L (m)':>10} | {'RMSE_R (m)':>10} | {'MAE_L (m)':>10} | {'MAE_R (m)':>10}"
    print("\n" + "=" * len(header))
    print(header)
    print("=" * len(header))
    for name, m in results.items():
        print(
            f"{name:<8} | {m['rmse_L']:>10.4f} | {m['rmse_R']:>10.4f} "
            f"| {m['mae_L']:>10.4f} | {m['mae_R']:>10.4f}"
        )
    print("=" * len(header))
