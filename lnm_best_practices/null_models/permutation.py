"""Permutation-based inference for Lesion Network Mapping.

Implements symptom label permutation with maximum-statistic FWER
correction, following lnm_nulls/lnm_compute.m (Zalesky et al.).

The maximum-statistic approach provides strong control of FWER: for each
permutation the maximum test statistic across all parcels is stored, and
the empirical statistic at each parcel is compared against this null
distribution of maxima.

References:
    lnm_nulls/lnm_compute.m - Original MATLAB implementation
    Nichols & Holmes (2002). Nonparametric permutation tests.
"""

from __future__ import annotations

from typing import Optional, Tuple

import numpy as np
from numpy.typing import NDArray


class PermutationTest:
    """Permutation-based inference for lesion network mapping.

    Uses symptom label permutation to build a null distribution of LNM
    maps, then applies the maximum-statistic method for strong FWER
    control.

    Follows lnm_compute.m:
        for i=1:Perms
            lnm_null = ind_lesion_null' * w / K;
            null_dist(i) = max(lnm_null);  % max statistic for FWER
        end

    Parameters
    ----------
    random_state : int or None, optional
        Random seed for reproducibility.
    """

    def __init__(self, random_state: Optional[int] = None) -> None:
        self.random_state = random_state
        self._rng = np.random.default_rng(random_state)

    def permute_symptoms(
        self,
        symptoms: NDArray[np.floating],
        n_permutations: int = 1000,
    ) -> NDArray[np.floating]:
        """Generate permuted symptom label vectors.

        For each permutation the symptom labels are randomly shuffled
        across subjects, breaking any association between lesion location
        and symptom severity.

        Parameters
        ----------
        symptoms : ndarray of shape (n_subjects,)
            Original symptom scores.
        n_permutations : int, optional
            Number of permutations. Default is 1000.

        Returns
        -------
        permuted : ndarray of shape (n_permutations, n_subjects)
            Matrix where each row is a random permutation of the
            original symptom vector.
        """
        symptoms = np.asarray(symptoms, dtype=np.float64)
        n_subjects = len(symptoms)
        permuted = np.zeros((n_permutations, n_subjects), dtype=np.float64)

        for i in range(n_permutations):
            permuted[i] = self._rng.permutation(symptoms)

        return permuted

    @staticmethod
    def compute_max_statistic(
        null_maps: NDArray[np.floating],
    ) -> NDArray[np.floating]:
        """Compute the maximum statistic for each permutation.

        Follows lnm_compute.m: null_dist(i) = max(lnm_null)

        Parameters
        ----------
        null_maps : ndarray of shape (n_permutations, n_parcels)
            Null LNM maps.

        Returns
        -------
        max_stats : ndarray of shape (n_permutations,)
            Maximum absolute value in each null map.
        """
        null_maps = np.asarray(null_maps)
        return np.max(np.abs(null_maps), axis=1)

    @staticmethod
    def fwer_correction(
        empirical_map: NDArray[np.floating],
        null_distribution: NDArray[np.floating],
        alpha: float = 0.05,
    ) -> Tuple[NDArray[np.floating], NDArray[np.bool_]]:
        """Apply FWER correction using the maximum-statistic method.

        Follows lnm_compute.m lines 64-66:
            null_dist_srt = sort(null_dist);
            critical_value = null_dist_srt(ceil(0.95*Perms));
            ind_sig = lnm > critical_value;

        Parameters
        ----------
        empirical_map : ndarray of shape (n_parcels,)
            Observed LNM values per parcel.
        null_distribution : ndarray of shape (n_permutations, n_parcels)
            Null LNM maps.
        alpha : float, optional
            Significance level. Default is 0.05.

        Returns
        -------
        fwer_p_values : ndarray of shape (n_parcels,)
            FWER-corrected p-values for each parcel.
        significant : ndarray of shape (n_parcels,) (dtype=bool)
            Boolean mask indicating significant parcels.
        """
        empirical_map = np.asarray(empirical_map, dtype=np.float64)
        null_distribution = np.asarray(null_distribution, dtype=np.float64)
        n_perms = null_distribution.shape[0]

        # Maximum absolute statistic per permutation
        max_null = np.max(np.abs(null_distribution), axis=1)
        abs_empirical = np.abs(empirical_map)

        # FWER p-value: proportion of null maxima >= observed value
        fwer_p_values = np.zeros(len(empirical_map), dtype=np.float64)
        for j in range(len(empirical_map)):
            fwer_p_values[j] = np.mean(max_null >= abs_empirical[j])

        # Avoid p=0
        fwer_p_values = np.maximum(fwer_p_values, 1.0 / n_perms)

        significant = fwer_p_values < alpha

        return fwer_p_values, significant

    def run(
        self,
        lesion_matrix: NDArray[np.floating],
        connectome: NDArray[np.floating],
        symptoms: NDArray[np.floating],
        n_permutations: int = 1000,
        alpha: float = 0.05,
        seed: Optional[int] = None,
    ) -> "PermutationResult":
        """Run the complete permutation test pipeline.

        Follows lnm_compute.m:
            1. Compute empirical LNM
            2. For each permutation, shuffle symptoms and compute null LNM
            3. Apply max-statistic FWER correction

        Parameters
        ----------
        lesion_matrix : ndarray of shape (n_subjects, n_parcels)
            Binary lesion matrix.
        connectome : ndarray of shape (n_parcels, n_parcels)
            Connectivity matrix.
        symptoms : ndarray of shape (n_subjects,)
            Symptom scores.
        n_permutations : int
            Number of permutations.
        alpha : float
            Significance level.
        seed : int, optional
            Random seed.

        Returns
        -------
        PermutationResult
            Result object with empirical map, null distribution, p-values.
        """
        from ..core.lnm import compute_lnm, compute_slnm

        lesion_matrix = np.asarray(lesion_matrix, dtype=np.float64)
        connectome = np.asarray(connectome, dtype=np.float64)
        symptoms = np.asarray(symptoms, dtype=np.float64).ravel()

        if seed is not None:
            self._rng = np.random.default_rng(seed)

        # Compute empirical LNM
        empirical_map = compute_lnm(lesion_matrix, connectome)

        # Generate permuted symptoms
        permuted = self.permute_symptoms(symptoms, n_permutations)

        # Compute null LNM maps
        null_maps = np.zeros((n_permutations, lesion_matrix.shape[1]), dtype=np.float64)
        for i in range(n_permutations):
            null_maps[i] = compute_slnm(lesion_matrix, connectome, permuted[i])

        # FWER correction
        fwer_p, significant = self.fwer_correction(empirical_map, null_maps, alpha)

        return PermutationResult(
            empirical_map=empirical_map,
            null_distribution=null_maps,
            p_values=fwer_p,
            fwer_p_values=fwer_p,
            significant=significant,
        )


class PermutationResult:
    """Container for permutation test results.

    Attributes
    ----------
    empirical_map : ndarray of shape (n_parcels,)
        Observed LNM map.
    null_distribution : ndarray of shape (n_permutations, n_parcels)
        Null LNM maps from symptom permutation.
    p_values : ndarray of shape (n_parcels,)
        FWER-corrected p-values.
    fwer_p_values : ndarray of shape (n_parcels,)
        FWER-corrected p-values (alias for p_values).
    significant : ndarray of shape (n_parcels,) (dtype=bool)
        Boolean mask of significant parcels.
    """

    def __init__(
        self,
        empirical_map: NDArray[np.floating],
        null_distribution: NDArray[np.floating],
        p_values: NDArray[np.floating],
        fwer_p_values: NDArray[np.floating],
        significant: NDArray[np.bool_],
    ) -> None:
        self.empirical_map = empirical_map
        self.null_distribution = null_distribution
        self.p_values = p_values
        self.fwer_p_values = fwer_p_values
        self.significant = significant
