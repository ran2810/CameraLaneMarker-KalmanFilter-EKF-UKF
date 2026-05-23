"""
visualization.py — Plotting helpers for lane filter results.
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from typing import Optional, Dict, List


# ------------------------------------------------------------------
# Colour palette — readable on white and dark backgrounds
# ------------------------------------------------------------------
COLORS = {
    "truth":    "#2c2c2c",
    "meas":     "#aaaaaa",
    "KF":       "#1f77b4",   # blue
    "EKF":      "#d62728",   # red
    "UKF":      "#2ca02c",   # green
    "shade_KF": "#aec7e8",
    "shade_EKF":"#f5b0b0",
    "shade_UKF":"#98df8a",
}

PHASE_COLORS = ["#fffbe6", "#fff0e6", "#fce4ec", "#e8f5e9", "#e3f2fd"]
PHASE_LABELS = [
    "①  Straight A9",
    "②  Curve → Dreieck",
    "③  Transition road",
    "④  Joined A8",
    "⑤  S-curve A8",
]
PHASE_BOUNDS = [0, 30, 50, 70, 90, 120]


def _add_phase_bands(ax: plt.Axes, alpha: float = 0.18) -> None:
    """Shade scenario phases on an axes."""
    for i in range(5):
        ax.axvspan(
            PHASE_BOUNDS[i], PHASE_BOUNDS[i + 1],
            color=PHASE_COLORS[i], alpha=alpha, zorder=0
        )
        ax.text(
            0.5 * (PHASE_BOUNDS[i] + PHASE_BOUNDS[i + 1]),
            ax.get_ylim()[1] * 0.97,
            PHASE_LABELS[i],
            ha="center", va="top", fontsize=6.5, color="#555555",
            style="italic"
        )


def plot_c0_comparison(
    t:          np.ndarray,
    true_L:     np.ndarray,
    true_R:     np.ndarray,
    meas_L:     np.ndarray,
    meas_R:     np.ndarray,
    estimates:  Dict[str, Dict[str, np.ndarray]],
    title:      str = "C0 Lateral Offset — Left & Right Lane Markers",
    save_path:  Optional[str] = None,
) -> plt.Figure:
    """
    Plot C0 estimates from multiple filters versus ground truth.

    Args:
        t:           Timestamps [T]
        true_L/R:    Ground truth C0 left/right [T]
        meas_L/R:    Noisy measurements (NaN = dropout) [T]
        estimates:   {"KF": {"C0_L": [...], "C0_R": [...]}, "EKF": {...}, ...}
        title:       Figure suptitle
        save_path:   If given, save figure here.

    Returns:
        matplotlib Figure
    """
    fig, axes = plt.subplots(2, 1, figsize=(14, 8), sharex=True)

    for ax, side, true_C0, meas in zip(
        axes,
        ["Left", "Right"],
        [true_L, true_R],
        [meas_L, meas_R],
    ):
        key = "C0_L" if side == "Left" else "C0_R"

        # Measurements
        valid = ~np.isnan(meas)
        ax.scatter(
            t[valid], meas[valid],
            s=3, color=COLORS["meas"], alpha=0.5, zorder=1, label="Camera (noisy)"
        )

        # Ground truth
        ax.plot(t, true_C0, color=COLORS["truth"], lw=1.5, zorder=3, label="Ground truth")

        # Filter estimates
        for name, est in estimates.items():
            ax.plot(
                t, est[key],
                color=COLORS.get(name, "purple"),
                lw=1.4, zorder=4, label=name,
                linestyle="--" if name == "KF" else "-"
            )

        ax.set_ylabel(f"C0 {side} (m)", fontsize=10)
        ax.legend(loc="upper right", fontsize=8, ncol=2)
        ax.grid(True, alpha=0.3)
        ax.set_ylim(true_C0.min() - 0.25, true_C0.max() + 0.25)

    _add_phase_bands(axes[0])
    _add_phase_bands(axes[1])

    axes[1].set_xlabel("Time (s)", fontsize=10)
    fig.suptitle(title, fontsize=12, fontweight="bold")
    fig.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"Saved → {save_path}")

    return fig


def plot_state_components(
    t:         np.ndarray,
    true_C1:   np.ndarray,
    true_C2:   np.ndarray,
    true_C3:   np.ndarray,
    estimates: Dict[str, Dict[str, np.ndarray]],
    save_path: Optional[str] = None,
) -> plt.Figure:
    """
    Plot estimated C1, C2, C3 vs ground truth for each filter.
    """
    fig, axes = plt.subplots(3, 1, figsize=(14, 9), sharex=True)
    labels = {"C1": "C1  Heading (rad)", "C2": "C2  Curvature (1/m)", "C3": "C3  Curv rate (1/m²)"}
    truths = {"C1": true_C1, "C2": true_C2, "C3": true_C3}

    for ax, (key, ylabel) in zip(axes, labels.items()):
        ax.plot(t, truths[key], color=COLORS["truth"], lw=1.5, label="Ground truth")
        for name, est in estimates.items():
            if key in est:
                ax.plot(
                    t, est[key],
                    color=COLORS.get(name, "purple"), lw=1.3,
                    linestyle="--" if name == "KF" else "-", label=name
                )
        ax.set_ylabel(ylabel, fontsize=9)
        ax.legend(loc="upper right", fontsize=8)
        ax.grid(True, alpha=0.3)

    axes[-1].set_xlabel("Time (s)", fontsize=10)
    fig.suptitle("Lane Polynomial Coefficients C1 / C2 / C3", fontsize=12, fontweight="bold")
    fig.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"Saved → {save_path}")

    return fig


def plot_nis(
    t:        np.ndarray,
    nis_dict: Dict[str, np.ndarray],
    obs_dim:  int = 1,
    save_path: Optional[str] = None,
) -> plt.Figure:
    """Plot NIS sequences with chi² consistency bounds."""
    from scipy.stats import chi2 as chi2_dist

    fig, ax = plt.subplots(figsize=(14, 4))

    # Expected value and 95% per-step bounds (chi²(m))
    lo = chi2_dist.ppf(0.025, df=obs_dim)
    hi = chi2_dist.ppf(0.975, df=obs_dim)
    ax.axhline(obs_dim, color="black", lw=1, ls="--", label=f"E[NIS] = {obs_dim}")
    ax.axhspan(lo, hi, alpha=0.1, color="gray", label="95% chi² band")

    for name, nis_vals in nis_dict.items():
        ax.plot(t, nis_vals, color=COLORS.get(name, "purple"), lw=0.8,
                alpha=0.85, label=name)

    ax.set_xlabel("Time (s)", fontsize=10)
    ax.set_ylabel("NIS", fontsize=10)
    ax.set_title("Normalised Innovation Squared (filter consistency)", fontsize=11)
    ax.set_ylim(0, min(chi2_dist.ppf(0.999, df=obs_dim) * 3, 20))
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")

    return fig


def plot_rmse_per_phase(
    phase_rmse: Dict[str, Dict[int, float]],
    phase_labels: Optional[List[str]] = None,
    save_path: Optional[str] = None,
) -> plt.Figure:
    """Bar chart of RMSE per filter per scenario phase."""
    if phase_labels is None:
        phase_labels = [f"Phase {i}" for i in range(1, 6)]

    filters = list(phase_rmse.keys())
    n_phases = len(phase_labels)
    x = np.arange(n_phases)
    width = 0.25

    fig, ax = plt.subplots(figsize=(11, 5))

    for j, name in enumerate(filters):
        vals = [phase_rmse[name].get(i + 1, 0.0) for i in range(n_phases)]
        ax.bar(x + j * width, vals, width, label=name,
               color=COLORS.get(name, "purple"), alpha=0.85, edgecolor="white")

    ax.set_xticks(x + width)
    ax.set_xticklabels(phase_labels, fontsize=8)
    ax.set_ylabel("RMSE C0_L (m)", fontsize=10)
    ax.set_title("Per-phase C0 RMSE (left marker)", fontsize=11)
    ax.legend(fontsize=9)
    ax.grid(True, axis="y", alpha=0.3)
    fig.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")

    return fig
