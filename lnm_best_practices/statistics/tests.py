"""
Statistical tests for LNM analysis.

Provides common statistical tests used in lesion-symptom mapping.

References:
    NiiStat/nii_stat_core.m - GLM implementation
"""

import numpy as np
from scipy import stats
from typing import Optional, Tuple


def one_sample_t_test(
    data: np.ndarray,
    null_mean: float = 0.0,
) -> Tuple[float, float]:
    """One-sample t-test.

    Parameters
    ----------
    data : np.ndarray
        Sample data (n,)
    null_mean : float, default=0.0
        Null hypothesis mean

    Returns
    -------
    tuple
        (t-statistic, p-value)
    """
    t_stat, p_value = stats.ttest_1samp(data, null_mean)
    return t_stat, p_value


def two_sample_t_test(
    group1: np.ndarray,
    group2: np.ndarray,
    equal_var: bool = True,
) -> Tuple[float, float]:
    """Two-sample t-test.

    Parameters
    ----------
    group1 : np.ndarray
        First group data
    group2 : np.ndarray
        Second group data
    equal_var : bool, default=True
        Assume equal variances (pooled variance)

    Returns
    -------
    tuple
        (t-statistic, p-value)
    """
    t_stat, p_value = stats.ttest_ind(group1, group2, equal_var=equal_var)
    return t_stat, p_value


def pearson_correlation(
    x: np.ndarray,
    y: np.ndarray,
) -> Tuple[float, float]:
    """Pearson correlation coefficient.

    Parameters
    ----------
    x : np.ndarray
        First variable
    y : np.ndarray
        Second variable

    Returns
    -------
    tuple
        (correlation, p-value)
    """
    r, p_value = stats.pearsonr(x, y)
    return r, p_value


def glm_t_test(
    X: np.ndarray,
    y: np.ndarray,
    contrast: Optional[np.ndarray] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """General linear model t-test.

    Implements the GLM t-test from NiiStat's glm_quick_t function.

    Parameters
    ----------
    X : np.ndarray
        Design matrix (n_subjects x n_regressors)
    y : np.ndarray
        Dependent variable (n_subjects,)
    contrast : np.ndarray, optional
        Contrast vector (n_regressors,). Default: [0, 1] for second regressor.

    Returns
    -------
    tuple
        (t-statistic, p-value) for each voxel/variable
    """
    n_subjects, n_regressors = X.shape

    if contrast is None:
        contrast = np.zeros(n_regressors)
        if n_regressors > 1:
            contrast[1] = 1.0
        else:
            contrast[0] = 1.0

    # Compute beta: (X'X)^-1 X'y
    try:
        XtX_inv = np.linalg.pinv(X.T @ X)
        beta = XtX_inv @ X.T @ y
    except np.linalg.LinAlgError:
        return np.nan, np.nan

    # Residuals
    residuals = y - X @ beta
    df = n_subjects - n_regressors

    if df <= 0:
        return np.nan, np.nan

    # Residual variance
    sigma2 = np.sum(residuals**2) / df

    # Variance of contrast
    c_var = sigma2 * (contrast @ XtX_inv @ contrast)

    if c_var <= 0:
        return np.nan, np.nan

    # t-statistic
    t_stat = (contrast @ beta) / np.sqrt(c_var)

    # p-value (two-tailed)
    p_value = 2 * stats.t.sf(np.abs(t_stat), df)

    return t_stat, p_value


def fishers_exact_test(
    contingency_table: np.ndarray,
) -> Tuple[float, float]:
    """Fisher's exact test for 2x2 contingency table.

    Used in VLSM (voxel-based lesion-symptom mapping).

    Parameters
    ----------
    contingency_table : np.ndarray
        2x2 contingency table

    Returns
    -------
    tuple
        (odds-ratio, p-value)
    """
    odds_ratio, p_value = stats.fisher_exact(contingency_table)
    return odds_ratio, p_value


def zscore_to_pvalue(z_scores: np.ndarray) -> np.ndarray:
    """Convert z-scores to two-tailed p-values.

    Parameters
    ----------
    z_scores : np.ndarray
        Z-score map

    Returns
    -------
    np.ndarray
        P-values
    """
    return 2 * stats.norm.sf(np.abs(z_scores))


def pvalue_to_zscore(p_values: np.ndarray) -> np.ndarray:
    """Convert two-tailed p-values to z-scores.

    Parameters
    ----------
    p_values : np.ndarray
        P-values

    Returns
    -------
    np.ndarray
        Z-scores (signed)
    """
    z = stats.norm.isf(p_values / 2)
    return z
