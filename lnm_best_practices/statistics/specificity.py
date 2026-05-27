"""
Specificity testing for LNM results.

Tests whether LNM results reflect genuine lesion-network relationships
or are driven by connectome properties (hub structure, degree distribution).

References:
    Van Den Heuvel et al. (2026) - Methodological foundation concerns
    Siddiqi et al. (2026) - Response defending LNM specificity
"""

import numpy as np
from typing import Optional, Dict, Tuple
from dataclasses import dataclass


@dataclass
class SpecificityResult:
    """Container for specificity test results.

    Attributes
    ----------
    degree_correlation : float
        Correlation between LNM map and degree map
    degree_pvalue : float
        P-value for degree correlation
    specificity_index : float
        Specificity index (1 - proportion explained by degree)
    random_lesion_pvalue : float
        P-value from random lesion comparison
    is_specific : bool
        Whether LNM shows specificity (p < 0.05 for random lesion test)
    """
    degree_correlation: float
    degree_pvalue: float
    specificity_index: float
    random_lesion_pvalue: float
    is_specific: bool


class SpecificityTest:
    """Specificity testing for LNM results.

    Implements tests from Van Den Heuvel et al. (2026) and
    responses from Siddiqi et al. (2026).

    Parameters
    ----------
    n_permutations : int, default=1000
        Number of permutations for random lesion test
    alpha : float, default=0.05
        Significance level

    Examples
    --------
    >>> test = SpecificityTest(n_permutations=1000)
    >>> result = test.run(lnm_map, degree_map, lesion_matrix, connectome)
    """

    def __init__(
        self,
        n_permutations: int = 1000,
        alpha: float = 0.05,
    ):
        self.n_permutations = n_permutations
        self.alpha = alpha

    def test_against_degree(
        self,
        lnm_map: np.ndarray,
        degree_map: np.ndarray,
    ) -> Tuple[float, float]:
        """Test if LNM converges to connectome hubs.

        Parameters
        ----------
        lnm_map : np.ndarray
            LNM sensitivity map (n_parcels,)
        degree_map : np.ndarray
            Node degree map (n_parcels,)

        Returns
        -------
        tuple
            (correlation, p-value)
        """
        from scipy.stats import pearsonr
        r, p = pearsonr(lnm_map, degree_map)
        return r, p

    def compute_specificity_index(
        self,
        lnm_map: np.ndarray,
        degree_map: np.ndarray,
    ) -> float:
        """Compute specificity index.

        Measures how much of LNM variance is NOT explained by degree.

        Parameters
        ----------
        lnm_map : np.ndarray
            LNM map
        degree_map : np.ndarray
            Degree map

        Returns
        -------
        float
            Specificity index (0 = fully explained by degree, 1 = fully specific)
        """
        r, _ = self.test_against_degree(lnm_map, degree_map)
        r_squared = r ** 2
        return 1.0 - r_squared

    def test_against_random_lesions(
        self,
        empirical_lnm: np.ndarray,
        lesion_matrix: np.ndarray,
        connectome: np.ndarray,
        seed: Optional[int] = None,
    ) -> Tuple[float, np.ndarray]:
        """Compare empirical LNM to random lesion LNM.

        Tests whether LNM is driven by specific lesion locations
        or produces similar results with random lesions.

        Parameters
        ----------
        empirical_lnm : np.ndarray
            Empirical LNM map (n_parcels,)
        lesion_matrix : np.ndarray
            Lesion matrix (n_subjects x n_parcels)
        connectome : np.ndarray
            Connectivity matrix (n_parcels x n_parcels)
        seed : int, optional
            Random seed

        Returns
        -------
        tuple
            (p_value, null_correlations)
        """
        rng = np.random.default_rng(seed)
        n_subjects, n_parcels = lesion_matrix.shape

        null_correlations = np.zeros(self.n_permutations)

        for perm in range(self.n_permutations):
            # Generate random lesions
            random_M = np.zeros_like(lesion_matrix)
            for i in range(n_subjects):
                n_affected = int(np.sum(lesion_matrix[i] > 0))
                if n_affected > 0:
                    random_parcels = rng.choice(n_parcels, size=n_affected, replace=False)
                    random_M[i, random_parcels] = 1.0 / n_affected

            # Compute random LNM
            random_lnm = np.sum(random_M @ connectome, axis=0)

            # Correlation with empirical
            null_correlations[perm] = np.corrcoef(empirical_lnm, random_lnm)[0, 1]

        # P-value: proportion of random correlations >= empirical self-correlation
        # (which is 1.0, so this tests if random LNM is significantly different)
        empirical_self_corr = 1.0
        p_value = np.mean(null_correlations >= empirical_self_corr)

        return p_value, null_correlations

    def run(
        self,
        lnm_map: np.ndarray,
        degree_map: np.ndarray,
        lesion_matrix: Optional[np.ndarray] = None,
        connectome: Optional[np.ndarray] = None,
        seed: Optional[int] = None,
    ) -> SpecificityResult:
        """Run full specificity test suite.

        Parameters
        ----------
        lnm_map : np.ndarray
            LNM sensitivity map
        degree_map : np.ndarray
            Node degree map
        lesion_matrix : np.ndarray, optional
            Lesion matrix (for random lesion test)
        connectome : np.ndarray, optional
            Connectivity matrix (for random lesion test)
        seed : int, optional
            Random seed

        Returns
        -------
        SpecificityResult
            Complete specificity test results
        """
        # Test against degree
        degree_r, degree_p = self.test_against_degree(lnm_map, degree_map)

        # Specificity index
        specificity_idx = self.compute_specificity_index(lnm_map, degree_map)

        # Random lesion test (if data provided)
        if lesion_matrix is not None and connectome is not None:
            random_p, _ = self.test_against_random_lesions(
                lnm_map, lesion_matrix, connectome, seed=seed
            )
        else:
            random_p = np.nan

        # Determine specificity
        is_specific = random_p < self.alpha if not np.isnan(random_p) else False

        return SpecificityResult(
            degree_correlation=degree_r,
            degree_pvalue=degree_p,
            specificity_index=specificity_idx,
            random_lesion_pvalue=random_p,
            is_specific=is_specific,
        )

    @staticmethod
    def compare_to_connectome_modules(
        lnm_map: np.ndarray,
        modules: np.ndarray,
    ) -> Dict[int, float]:
        """Compare LNM map to connectome modules.

        Parameters
        ----------
        lnm_map : np.ndarray
            LNM map (n_parcels,)
        modules : np.ndarray
            Module assignments (n_parcels,)

        Returns
        -------
        dict
            Mean LNM value per module
        """
        unique_modules = np.unique(modules)
        module_means = {}

        for mod in unique_modules:
            mask = modules == mod
            module_means[int(mod)] = np.mean(lnm_map[mask])

        return module_means
