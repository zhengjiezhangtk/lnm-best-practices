"""Topological null model for Lesion Network Mapping.

Generates random networks preserving degree distribution, testing whether
LNM results are driven by connectome structure rather than spatial lesion
placement.

Implements the degree-corrected stochastic block model (dcSBM) from
lnm_nulls/dcsbm.m (Zalesky et al.) and the Maslov-Sneppen edge-swapping
algorithm.

References:
    lnm_nulls/dcsbm.m - Original MATLAB dcSBM implementation
    lnm_nulls/precompute_topology_randomization.m - Pre-computation
"""

from __future__ import annotations

from typing import Optional

import numpy as np
from numpy.typing import NDArray


class TopologicalNullModel:
    """Topological null model for lesion network mapping.

    Generates null distributions by randomizing network topology while
    preserving the degree (strength) distribution.

    Two strategies:
    - ``dcsbm``: Degree-corrected stochastic block model (from dcsbm.m)
    - ``maslov_sneppen``: Edge-swapping algorithm

    Parameters
    ----------
    random_state : int or None, optional
        Random seed for reproducibility.
    """

    def __init__(self, random_state: Optional[int] = None) -> None:
        self.random_state = random_state
        self._rng = np.random.default_rng(random_state)

    def generate_null(
        self,
        connectome: NDArray[np.floating],
        n_permutations: int = 1000,
        lesion_matrix: Optional[NDArray[np.floating]] = None,
    ) -> NDArray[np.floating]:
        """Generate null connectomes with preserved degree distribution.

        Follows dcsbm.m: generates fully connected FC matrix with modules
        and log-normal degree distribution. Weights from beta distribution.

        Parameters
        ----------
        connectome : ndarray of shape (n_parcels, n_parcels)
            Observed connectivity matrix.
        n_permutations : int, optional
            Number of null networks. Default is 1000.
        lesion_matrix : ndarray of shape (n_subjects, n_parcels), optional
            If provided, null LNM maps are returned.

        Returns
        -------
        null_maps : ndarray of shape (n_permutations, n_parcels)
            Null LNM maps or node strength vectors.
        """
        connectome = np.asarray(connectome)
        n_parcels = connectome.shape[0]

        # Observed node strengths (like MATLAB's s=sum(w)/N)
        observed_strength = np.sum(connectome, axis=0) / n_parcels

        null_maps = np.zeros((n_permutations, n_parcels), dtype=np.float64)

        for perm_idx in range(n_permutations):
            # Generate dcSBM null network
            random_C = self._generate_dcsbm_null(n_parcels, observed_strength)

            if lesion_matrix is not None:
                # Compute LNM for each subject and average
                lnm_per_subject = lesion_matrix @ random_C
                null_maps[perm_idx] = np.mean(lnm_per_subject, axis=0)
            else:
                # Return node strengths of null network
                null_maps[perm_idx] = np.sum(random_C, axis=0) / n_parcels

        return null_maps

    def degree_preserving_randomization(
        self,
        connectome: NDArray[np.floating],
        n_swaps: int = 10000,
    ) -> NDArray[np.floating]:
        """Randomize a network preserving its exact degree sequence.

        Uses the Maslov-Sneppen edge-swapping algorithm.

        Parameters
        ----------
        connectome : ndarray of shape (n_parcels, n_parcels)
            Symmetric connectivity matrix.
        n_swaps : int, optional
            Number of edge-swap attempts. Default is 10000.

        Returns
        -------
        randomized : ndarray of shape (n_parcels, n_parcels)
            Randomized connectivity matrix.
        """
        connectome = np.asarray(connectome, dtype=np.float64)
        n = connectome.shape[0]

        # Work with upper-triangular binary adjacency
        upper = np.triu(connectome, k=1)
        binary = (upper > 0).astype(np.float64)

        # Collect edge list
        edge_rows, edge_cols = np.where(binary > 0)
        n_edges = len(edge_rows)

        if n_edges < 2:
            return connectome.copy()

        edge_list = list(zip(edge_rows.tolist(), edge_cols.tolist()))
        edge_set = set(edge_list)

        for _ in range(n_swaps):
            idx1 = self._rng.integers(0, n_edges)
            idx2 = self._rng.integers(0, n_edges)
            if idx1 == idx2:
                continue

            a, b = edge_list[idx1]
            c, d = edge_list[idx2]

            if a > b:
                a, b = b, a
            if c > d:
                c, d = d, c

            # Try swap: (a,b), (c,d) -> (a,d), (c,b) or (a,c), (b,d)
            if self._rng.random() < 0.5:
                new1 = (min(a, d), max(a, d))
                new2 = (min(c, b), max(c, b))
            else:
                new1 = (min(a, c), max(a, c))
                new2 = (min(b, d), max(b, d))

            if new1[0] == new1[1] or new2[0] == new2[1]:
                continue
            if new1 in edge_set or new2 in edge_set:
                continue
            if new1 == new2:
                continue

            edge_set.discard((a, b))
            edge_set.discard((c, d))
            edge_set.add(new1)
            edge_set.add(new2)
            edge_list[idx1] = new1
            edge_list[idx2] = new2

        # Reconstruct adjacency matrix
        randomized = np.zeros((n, n), dtype=np.float64)
        for i, j in edge_list:
            randomized[i, j] = connectome[i, j]
            randomized[j, i] = connectome[j, i]

        return randomized

    def compute_pvalue(
        self,
        empirical_map: NDArray[np.floating],
        null_distribution: NDArray[np.floating],
    ) -> NDArray[np.floating]:
        """Compute permutation p-values for each parcel.

        Parameters
        ----------
        empirical_map : ndarray of shape (n_parcels,)
            Observed LNM values per parcel.
        null_distribution : ndarray of shape (n_permutations, n_parcels)
            Null LNM maps.

        Returns
        -------
        p_values : ndarray of shape (n_parcels,)
            Permutation p-values.
        """
        empirical_map = np.asarray(empirical_map)
        null_distribution = np.asarray(null_distribution)
        n_perms = null_distribution.shape[0]

        exceedances = np.sum(
            null_distribution >= empirical_map[np.newaxis, :], axis=0
        )
        p_values = (exceedances + 1) / (n_perms + 1)

        return p_values

    def _generate_dcsbm_null(
        self,
        n_parcels: int,
        target_strength: NDArray[np.floating],
    ) -> NDArray[np.floating]:
        """Generate a random network via the dcSBM approach.

        Follows dcsbm.m exactly:
        1. Log-normal strength distribution: theta = lognrnd(1, sigma, N, 1)
        2. Connection probabilities: p = ones(N,N) * Bintra (no modules)
        3. Expected weights: mu = theta * theta' .* p
        4. Sample weights: w = betarnd(mu*k, (1-mu)*k)
        5. Symmetrize: w = triu(w,1) + triu(w,1)'

        Parameters
        ----------
        n_parcels : int
            Number of nodes.
        target_strength : ndarray of shape (n_parcels,)
            Observed node strength distribution to match.

        Returns
        -------
        w : ndarray of shape (n_parcels, n_parcels)
            Random symmetric connectivity matrix.
        """
        # Log-normal strength distribution (dcsbm.m line 41)
        # sigma controls heterogeneity: small=homogeneous, large=strong hubs
        log_target = np.log(target_strength + 1e-10)
        mu_log = np.mean(log_target)
        sigma_log = max(np.std(log_target), 0.2)
        theta = self._rng.lognormal(mu_log, sigma_log, size=n_parcels)

        # Connection probabilities: all equal (no modular structure)
        # Follows dcsbm.m: p=ones(N,N)*Bintra; p=p-diag(diag(p))
        p = np.ones((n_parcels, n_parcels))
        np.fill_diagonal(p, 0)

        # Expected weights: mu = theta * theta' .* p (dcsbm.m line 61)
        mu = np.outer(theta, theta) * p
        mu = mu / (np.max(mu) + np.finfo(float).eps)

        # Dispersion parameter for Beta distribution (dcsbm.m line 64)
        # Large k: weights tightly coupled around mu
        # Small k: high variability
        k = 10.0

        # Sample weights from Beta distribution (dcsbm.m line 68)
        alpha_param = np.clip(mu * k, 1e-6, None)
        beta_param = np.clip((1 - mu) * k, 1e-6, None)
        w = self._rng.beta(alpha_param, beta_param)

        # Symmetrize (dcsbm.m line 71)
        w = np.triu(w, k=1)
        w = w + w.T

        # Rescale to match mean observed strength
        current_strength = np.mean(np.sum(w, axis=0) / n_parcels)
        target_mean = np.mean(target_strength)
        if current_strength > 0:
            w = w * (target_mean / current_strength)

        return w
