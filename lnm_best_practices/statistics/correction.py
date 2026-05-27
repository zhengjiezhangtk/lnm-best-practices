"""
Multiple comparison correction methods for LNM.

Implements Bonferroni, FDR, and FWER correction methods.

References:
    NiiStat/fdr_bh.m - Benjamini-Hochberg FDR
    lnm_nulls - FWER via max-statistic
"""

import numpy as np
from typing import Tuple, Optional


def bonferroni_correction(
    p_values: np.ndarray,
    alpha: float = 0.05,
) -> Tuple[np.ndarray, np.ndarray]:
    """Bonferroni correction for multiple comparisons.

    Parameters
    ----------
    p_values : np.ndarray
        Uncorrected p-values
    alpha : float, default=0.05
        Family-wise error rate

    Returns
    -------
    tuple
        (corrected_p_values, significant_mask)
    """
    n_tests = len(p_values)
    corrected = np.minimum(p_values * n_tests, 1.0)
    significant = corrected < alpha
    return corrected, significant


def fdr_correction(
    p_values: np.ndarray,
    alpha: float = 0.05,
    method: str = 'bh',
) -> Tuple[np.ndarray, np.ndarray]:
    """False Discovery Rate correction.

    Implements Benjamini-Hochberg and Benjamini-Yekutieli methods.

    Parameters
    ----------
    p_values : np.ndarray
        Uncorrected p-values
    alpha : float, default=0.05
        FDR level
    method : str, default='bh'
        'bh' for Benjamini-Hochberg, 'by' for Benjamini-Yekutieli

    Returns
    -------
    tuple
        (corrected_p_values, significant_mask)
    """
    n_tests = len(p_values)

    # Sort p-values
    sorted_indices = np.argsort(p_values)
    sorted_p = p_values[sorted_indices]

    # Compute thresholds
    if method == 'bh':
        # Benjamini-Hochberg
        thresholds = alpha * np.arange(1, n_tests + 1) / n_tests
    elif method == 'by':
        # Benjamini-Yekutieli (for dependent tests)
        c_n = np.sum(1.0 / np.arange(1, n_tests + 1))
        thresholds = alpha * np.arange(1, n_tests + 1) / (n_tests * c_n)
    else:
        raise ValueError(f"Unknown method: {method}")

    # Find the largest k where p(k) <= threshold(k)
    below_threshold = sorted_p <= thresholds

    if not np.any(below_threshold):
        # No significant results
        corrected = np.ones(n_tests)
        significant = np.zeros(n_tests, dtype=bool)
    else:
        # Step-up procedure
        max_k = np.max(np.where(below_threshold)[0])

        # Adjust p-values
        corrected_sorted = np.ones(n_tests)
        for i in range(n_tests):
            if i <= max_k:
                corrected_sorted[i] = min(sorted_p[i] * n_tests / (i + 1), 1.0)
            else:
                # Propagate the minimum
                corrected_sorted[i] = min(corrected_sorted[i - 1], sorted_p[i] * n_tests / (i + 1))

        # Restore original order
        corrected = np.zeros(n_tests)
        corrected[sorted_indices] = corrected_sorted

        # Significant if corrected p < alpha
        significant = corrected < alpha

    return corrected, significant


def fwer_correction(
    null_distribution: np.ndarray,
    empirical_values: np.ndarray,
    alpha: float = 0.05,
) -> Tuple[np.ndarray, np.ndarray]:
    """Family-Wise Error Rate correction using max-statistic.

    Parameters
    ----------
    null_distribution : np.ndarray
        Null values (n_permutations x n_tests)
    empirical_values : np.ndarray
        Empirical values (n_tests,)
    alpha : float, default=0.05
        FWER level

    Returns
    -------
    tuple
        (fwer_p_values, significant_mask)
    """
    n_perms = null_distribution.shape[0]

    # Max statistic per permutation
    max_null = np.max(np.abs(null_distribution), axis=1)
    abs_empirical = np.abs(empirical_values)

    # FWER p-values
    fwer_p = np.zeros(len(empirical_values))
    for j in range(len(empirical_values)):
        fwer_p[j] = np.mean(max_null >= abs_empirical[j])

    fwer_p = np.maximum(fwer_p, 1.0 / n_perms)
    significant = fwer_p < alpha

    return fwer_p, significant


def tfce_enhancement(
    stat_map: np.ndarray,
    connectivity: Optional[np.ndarray] = None,
    E: float = 0.5,
    H: float = 2.0,
    n_steps: int = 100,
) -> np.ndarray:
    """Threshold-Free Cluster Enhancement.

    Parameters
    ----------
    stat_map : np.ndarray
        Statistical map (n_parcels,)
    connectivity : np.ndarray, optional
        Connectivity matrix (n_parcels x n_parcels)
    E : float, default=0.5
        Extent parameter
    H : float, default=2.0
        Height parameter
    n_steps : int, default=100
        Number of threshold steps

    Returns
    -------
    np.ndarray
        TFCE-enhanced map
    """
    if connectivity is None:
        # Use identity (no spatial clustering)
        connectivity = np.eye(len(stat_map))

    tfce_map = np.zeros_like(stat_map)
    abs_map = np.abs(stat_map)

    # Threshold range
    min_thresh = 0
    max_thresh = np.max(abs_map)
    thresholds = np.linspace(min_thresh, max_thresh, n_steps + 1)[1:]

    for thresh in thresholds:
        # Find supra-threshold clusters
        supra = abs_map >= thresh

        if not np.any(supra):
            continue

        # Find connected components
        from scipy.sparse.csgraph import connected_components
        from scipy.sparse import csr_matrix

        # Create subgraph of supra-threshold voxels
        sub_conn = connectivity[np.ix_(supra, supra)]
        n_components, labels = connected_components(csr_matrix(sub_conn), directed=False)

        # Enhance each component
        for comp in range(n_components):
            comp_mask = labels == comp
            comp_voxels = np.where(supra)[0][comp_mask]

            # Extent: number of voxels in cluster
            extent = len(comp_voxels)

            # Height: threshold value
            height = thresh

            # TFCE score
            score = (extent ** E) * (height ** H)

            # Add to TFCE map
            tfce_map[comp_voxels] += score

    # Preserve sign
    tfce_map *= np.sign(stat_map)

    return tfce_map
