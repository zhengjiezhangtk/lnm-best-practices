"""
Statistical Plots
-----------------
Charting utilities for Lesion Network Mapping (LNM) analysis results:
null distributions, scatter correlations, p-value maps, and specificity tests.

Dependencies
------------
- matplotlib >= 3.5
- numpy
- scipy (optional, for KDE overlay)
"""

from __future__ import annotations

from typing import Optional, Sequence, Tuple, Union

import matplotlib.pyplot as plt
import numpy as np


# ---------------------------------------------------------------------------
# Null distribution histogram
# ---------------------------------------------------------------------------

def plot_null_distribution(
    empirical_value: float,
    null_distribution: np.ndarray,
    title: str = "Null Distribution",
    xlabel: str = "Correlation",
    ylabel: str = "Frequency",
    num_bins: int = 50,
    alpha: float = 0.05,
    show_kde: bool = True,
    figsize: Tuple[int, int] = (8, 5),
    ax: Optional[plt.Axes] = None,
) -> plt.Figure:
    """Plot a null distribution histogram with the empirical value marked.

    Parameters
    ----------
    empirical_value : float
        The observed statistic to highlight.
    null_distribution : np.ndarray
        1-D array of null/permutation values.
    title : str
        Figure title.
    xlabel, ylabel : str
        Axis labels.
    num_bins : int
        Number of histogram bins.
    alpha : float
        Significance level; the corresponding critical value is drawn as a
        vertical dashed line.
    show_kde : bool
        Overlay a kernel-density estimate (requires ``scipy``).
    figsize : tuple of int
        Figure size.
    ax : matplotlib.axes.Axes or None
        Target axes.  A new figure is created when ``None``.

    Returns
    -------
    matplotlib.figure.Figure
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)
    else:
        fig = ax.figure

    counts, bins, patches = ax.hist(
        null_distribution,
        bins=num_bins,
        density=True,
        alpha=0.6,
        color="#4C72B0",
        edgecolor="white",
        linewidth=0.8,
        label="Null distribution",
    )

    # Optional KDE overlay
    if show_kde:
        try:
            from scipy.stats import gaussian_kde

            kde = gaussian_kde(null_distribution)
            x_grid = np.linspace(
                np.min(null_distribution), np.max(null_distribution), 300
            )
            ax.plot(x_grid, kde(x_grid), color="#C44E52", linewidth=1.5,
                    label="KDE")
        except ImportError:
            pass  # scipy not available; skip KDE

    # Empirical value
    ax.axvline(
        empirical_value,
        color="red",
        linewidth=2,
        linestyle="-",
        label=f"Empirical = {empirical_value:.4f}",
    )

    # Critical threshold (two-tailed)
    crit_low = float(np.percentile(null_distribution, 100 * alpha / 2))
    crit_high = float(np.percentile(null_distribution, 100 * (1 - alpha / 2)))
    for crit, side in [(crit_low, "lower"), (crit_high, "upper")]:
        ax.axvline(
            crit,
            color="grey",
            linewidth=1.5,
            linestyle="--",
            label=f"{side} {alpha*100:.0f}% = {crit:.4f}",
        )

    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.legend(fontsize=8, loc="upper right")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    fig.tight_layout()
    return fig


# ---------------------------------------------------------------------------
# Scatter correlation
# ---------------------------------------------------------------------------

def plot_correlation(
    x: np.ndarray,
    y: np.ndarray,
    xlabel: str = "X",
    ylabel: str = "Y",
    title: str = "Correlation",
    show_reg: bool = True,
    figsize: Tuple[int, int] = (7, 6),
    ax: Optional[plt.Axes] = None,
) -> Tuple[plt.Figure, float]:
    """Scatter plot of *x* vs *y* with Pearson correlation annotation.

    Points where either *x* or *y* is NaN or exactly zero are excluded.

    Parameters
    ----------
    x, y : np.ndarray
        Equal-length 1-D arrays.
    xlabel, ylabel : str
        Axis labels.
    title : str
        Figure title.
    show_reg : bool
        Overlay a least-squares regression line.
    figsize : tuple of int
        Figure size.
    ax : matplotlib.axes.Axes or None
        Target axes.  A new figure is created when ``None``.

    Returns
    -------
    fig : matplotlib.figure.Figure
    r : float
        Pearson correlation coefficient (``NaN`` if no valid data).
    """
    x = np.asarray(x, dtype=float).ravel()
    y = np.asarray(y, dtype=float).ravel()

    # Remove NaN / zero pairs
    mask = ~(np.isnan(x) | np.isnan(y) | (x == 0) | (y == 0))
    x, y = x[mask], y[mask]

    if len(x) == 0:
        return plt.figure(), float("nan")

    r = float(np.corrcoef(x, y)[0, 1])

    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)
    else:
        fig = ax.figure

    ax.scatter(x, y, s=18, c="#4C72B0", alpha=0.7, edgecolors="white",
               linewidth=0.5)

    if show_reg:
        m, b = np.polyfit(x, y, 1)
        xs = np.linspace(x.min(), x.max(), 100)
        ax.plot(xs, m * xs + b, color="#C44E52", linewidth=1.5)

    # Annotate r
    ax.text(
        0.05, 0.95,
        f"r = {r:.3f}",
        transform=ax.transAxes,
        fontsize=12,
        verticalalignment="top",
        bbox=dict(boxstyle="round,pad=0.3", facecolor="wheat", alpha=0.5),
    )

    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    fig.tight_layout()
    return fig, r


# ---------------------------------------------------------------------------
# P-value map
# ---------------------------------------------------------------------------

def plot_pvalue_map(
    p_values: np.ndarray,
    threshold: float = 0.05,
    title: str = "P-value Distribution",
    figsize: Tuple[int, int] = (10, 4),
    ax: Optional[plt.Axes] = None,
) -> plt.Figure:
    """Visualise a vector of p-values as a bar/heatmap.

    Bars whose p-value is below *threshold* are highlighted in red.

    Parameters
    ----------
    p_values : np.ndarray
        1-D array of p-values.
    threshold : float
        Significance threshold.
    title : str
        Figure title.
    figsize : tuple of int
        Figure size.
    ax : matplotlib.axes.Axes or None
        Target axes.

    Returns
    -------
    matplotlib.figure.Figure
    """
    p_values = np.asarray(p_values, dtype=float).ravel()
    indices = np.arange(len(p_values))

    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)
    else:
        fig = ax.figure

    colors = np.where(p_values < threshold, "#C44E52", "#4C72B0")
    ax.bar(indices, p_values, color=colors, width=1.0, edgecolor="none")
    ax.axhline(threshold, color="black", linestyle="--", linewidth=1,
               label=f"alpha = {threshold}")
    ax.set_xlabel("Parcel / Voxel index")
    ax.set_ylabel("p-value")
    ax.set_title(title)
    ax.legend(fontsize=9)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    fig.tight_layout()
    return fig


# ---------------------------------------------------------------------------
# Specificity test results
# ---------------------------------------------------------------------------

def plot_specificity_results(
    results: dict,
    figsize: Tuple[int, int] = (10, 6),
    ax: Optional[plt.Axes] = None,
) -> plt.Figure:
    """Bar chart of specificity-test results.

    *results* is expected to be a mapping such as::

        {
            "region_A": {"effect": 0.45, "p_value": 0.01},
            "region_B": {"effect": 0.12, "p_value": 0.38},
            ...
        }

    Parameters
    ----------
    results : dict
        Keys are region names; values are dicts with ``'effect'`` and
        ``'p_value'`` entries.
    figsize : tuple of int
        Figure size.
    ax : matplotlib.axes.Axes or None
        Target axes.

    Returns
    -------
    matplotlib.figure.Figure
    """
    regions = list(results.keys())
    effects = [v.get("effect", 0.0) for v in results.values()]
    pvals = [v.get("p_value", 1.0) for v in results.values()]

    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)
    else:
        fig = ax.figure

    x_pos = np.arange(len(regions))
    colors = ["#C44E52" if p < 0.05 else "#4C72B0" for p in pvals]

    bars = ax.bar(x_pos, effects, color=colors, edgecolor="white", width=0.6)

    # Annotate p-values above bars
    for bar, p in zip(bars, pvals):
        y_loc = bar.get_height()
        label = f"p={p:.3f}" if p >= 0.001 else "p<0.001"
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            y_loc,
            label,
            ha="center",
            va="bottom",
            fontsize=8,
        )

    ax.set_xticks(x_pos)
    ax.set_xticklabels(regions, rotation=45, ha="right")
    ax.set_ylabel("Effect size")
    ax.set_title("Specificity Test Results")
    ax.axhline(0, color="grey", linewidth=0.8)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    # Legend
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor="#C44E52", label="p < 0.05"),
        Patch(facecolor="#4C72B0", label="p >= 0.05"),
    ]
    ax.legend(handles=legend_elements, fontsize=9)

    fig.tight_layout()
    return fig
