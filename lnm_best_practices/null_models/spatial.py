"""Spatial null model for Lesion Network Mapping.

Randomizes lesion locations while preserving lesion size, following the
algorithm from lnm_nulls/lesion_assignment.m (Zalesky et al.).

Three assignment schemes:
    Type 1: Each parcel can be assigned at most one lesion (no overlap)
    Type 2: Parcels may be assigned multiple lesions
    Type 3: Like Type 2, but approximately preserves node strength

References:
    lnm_nulls/lesion_assignment.m - Original MATLAB implementation
    lnm_nulls/lnm_compute.m - LNM computation with null models
"""

from __future__ import annotations

from typing import Optional

import numpy as np
from numpy.typing import NDArray


class SpatialNullModel:
    """Spatial null model for lesion network mapping.

    Generates null distributions by randomizing lesion positions across
    brain parcels while preserving the number of lesions per subject.

    Follows the algorithm from lnm_nulls/lesion_assignment.m:
        - Assignment_Type==1: unique parcels (no overlap)
        - Assignment_Type==2: allow overlap
        - Assignment_Type==3: strength-preserving randomization

    Parameters
    ----------
    random_state : int or None, optional
        Random seed for reproducibility. Default is ``None``.
    """

    def __init__(self, random_state: Optional[int] = None) -> None:
        self.random_state = random_state
        self._rng = np.random.default_rng(random_state)

    def generate_null(
        self,
        lesion_matrix: NDArray[np.floating],
        n_permutations: int = 1000,
        method: str = "allow_overlap",
        strength: Optional[NDArray[np.floating]] = None,
        n_neighbors: int = 0,
    ) -> NDArray[np.floating]:
        """Generate null lesion maps by randomizing parcel assignments.

        Follows lesion_assignment.m with alpha=0 (random assignment).

        Parameters
        ----------
        lesion_matrix : ndarray of shape (n_subjects, n_parcels)
            Binary or weighted lesion matrix.
        n_permutations : int, optional
            Number of null permutations. Default is 1000.
        method : {"unique", "allow_overlap", "strength_preserved"}, optional
            Parcel assignment strategy:
            - "unique": Assignment_Type==1, each parcel gets at most one lesion
            - "allow_overlap": Assignment_Type==2, parcels may receive multiple
            - "strength_preserved": Assignment_Type==3, strength-matched sampling
        strength : ndarray of shape (n_parcels,), optional
            Node strength vector. Required for "strength_preserved".
        n_neighbors : int, optional
            Number of strength-matched neighbors. If 0, uses top 20%.

        Returns
        -------
        null_maps : ndarray of shape (n_permutations, n_parcels)
            Null LNM maps averaged across subjects.
        """
        lesion_matrix = np.asarray(lesion_matrix)
        n_subjects, n_parcels = lesion_matrix.shape

        # Precompute per-subject lesion counts (like MATLAB's K)
        lesion_counts = np.sum(lesion_matrix > 0, axis=1).astype(int)
        if np.any(lesion_counts > n_parcels):
            raise ValueError(
                f"A subject has {int(np.max(lesion_counts))} lesioned parcels "
                f"but only {n_parcels} parcels exist."
            )

        # Precompute strength-matched neighbors for Assignment_Type==3
        # Follows lnm_compute.m lines 36-42
        neighbors: Optional[NDArray[np.intp]] = None
        if method == "strength_preserved":
            if strength is None:
                raise ValueError(
                    "Node 'strength' vector is required for "
                    "method='strength_preserved'."
                )
            if n_neighbors <= 0:
                # Follows lnm_compute.m: U=0.2; un=ceil(U*(N-1))
                n_neighbors = max(1, int(np.ceil(0.2 * (n_parcels - 1))))
            neighbors = self._compute_strength_neighbors(strength, n_neighbors)

        null_maps = np.zeros((n_permutations, n_parcels), dtype=np.float64)

        for perm_idx in range(n_permutations):
            perm_map = np.zeros((n_subjects, n_parcels), dtype=np.float64)
            for subj_idx in range(n_subjects):
                k = lesion_counts[subj_idx]
                if k == 0:
                    continue

                if method == "unique":
                    # Assignment_Type==1: ind_lesion(randperm(N,K))=1
                    selected = self._rng.choice(n_parcels, size=k, replace=False)
                elif method == "allow_overlap":
                    # Assignment_Type==2: idx=randi(N,K,1); accumarray
                    selected = self._rng.integers(0, n_parcels, size=k)
                elif method == "strength_preserved":
                    # Assignment_Type==3: strength-matched sampling
                    assert neighbors is not None
                    selected = self._strength_preserved_sample(
                        lesion_matrix[subj_idx], neighbors, n_neighbors
                    )
                else:
                    raise ValueError(
                        f"Unknown method '{method}'. Expected one of: "
                        "'unique', 'allow_overlap', 'strength_preserved'."
                    )
                perm_map[subj_idx, selected] = 1.0

            # Average across subjects to get group-level null LNM
            null_maps[perm_idx] = perm_map.mean(axis=0)

        return null_maps

    def compute_pvalue(
        self,
        empirical_map: NDArray[np.floating],
        null_distribution: NDArray[np.floating],
    ) -> NDArray[np.floating]:
        """Compute permutation p-values for each parcel.

        Follows lnm_compute.m: p = (exceedances + 1) / (n_perms + 1)

        Parameters
        ----------
        empirical_map : ndarray of shape (n_parcels,)
            Observed LNM values per parcel.
        null_distribution : ndarray of shape (n_permutations, n_parcels)
            Null LNM maps.

        Returns
        -------
        p_values : ndarray of shape (n_parcels,)
            Permutation p-values for each parcel.
        """
        empirical_map = np.asarray(empirical_map)
        null_distribution = np.asarray(null_distribution)
        n_perms = null_distribution.shape[0]

        exceedances = np.sum(
            null_distribution >= empirical_map[np.newaxis, :], axis=0
        )
        # Conservative correction (adds 1 to numerator and denominator)
        p_values = (exceedances + 1) / (n_perms + 1)

        return p_values

    def _compute_strength_neighbors(
        self, strength: NDArray[np.floating], n_neighbors: int
    ) -> NDArray[np.intp]:
        """Find the top-n_neighbors strength-matched parcels per parcel.

        Follows lnm_compute.m lines 38-42:
            for i = 1:N
                [~, ind_srt] = sort(abs(s(i) - s));
                neighbors(i,:) = ind_srt(2:un+1);
            end

        Parameters
        ----------
        strength : ndarray of shape (n_parcels,)
            Node strength vector.
        n_neighbors : int
            Number of closest-strength neighbors.

        Returns
        -------
        neighbors : ndarray of shape (n_parcels, n_neighbors)
            Indices of the closest-strength neighbor parcels.
        """
        n_parcels = len(strength)
        n_neighbors = min(n_neighbors, n_parcels - 1)
        neighbors = np.zeros((n_parcels, n_neighbors), dtype=np.intp)
        for i in range(n_parcels):
            diff = np.abs(strength[i] - strength)
            diff[i] = np.inf  # exclude self
            sorted_idx = np.argsort(diff)
            neighbors[i] = sorted_idx[:n_neighbors]
        return neighbors

    def _strength_preserved_sample(
        self,
        lesion_vec: NDArray[np.floating],
        neighbors: NDArray[np.intp],
        n_neighbors: int,
    ) -> NDArray[np.intp]:
        """Sample parcels from strength-matched candidates.

        Follows lesion_assignment.m Assignment_Type==3:
            for i = 1:N
                k = ind_lesion_observed(i);
                if k > 0
                    picks = neighbors(i, randi(un, k, 1));
                    ind_lesion = ind_lesion + accumarray(picks', 1, [N,1]);
                end
            end

        Parameters
        ----------
        lesion_vec : ndarray of shape (n_parcels,)
            Observed lesion vector for a single subject.
        neighbors : ndarray of shape (n_parcels, n_neighbors)
            Precomputed strength-matched neighbor indices.
        n_neighbors : int
            Number of neighbors per parcel.

        Returns
        -------
        selected : ndarray of int
            Parcel indices selected for the null lesion.
        """
        lesioned_parcels = np.where(lesion_vec > 0)[0]
        selected_list: list[int] = []
        for p in lesioned_parcels:
            k = max(1, int(lesion_vec[p]))
            # randi(un, k, 1) -> random indices into neighbor list
            choices = self._rng.integers(0, n_neighbors, size=k)
            selected_list.extend(neighbors[p, choices].tolist())
        return np.array(selected_list, dtype=np.intp)
