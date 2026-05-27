"""Spin test for spatial autocorrelation-preserving null models.

Integrates BrainSpace's spin permutation test for parcellated
neuroimaging data, preserving spatial autocorrelation structure
when generating null distributions.

The spin test rotates parcellated data on the cortical surface sphere,
producing null maps that respect the spatial structure of the cortex.
This is critical for LNM analyses where spatial autocorrelation could
inflate false positive rates.

References:
    Vos de Wael et al. (2020). BrainSpace: a toolbox for the analysis
        of macroscale gradients in neuroimaging and connectomics.
        Communications Biology. https://doi.org/10.1038/s42003-020-0794-7
    Alexander-Bloch et al. (2018). On testing for spatial correspondence
        between maps. NeuroImage.
    lesionnetworkmapping/utils/spin_model.m - Original MATLAB implementation.
"""

from __future__ import annotations

from typing import Optional

import numpy as np
from numpy.typing import NDArray


class SpinTest:
    """Spin test for parcellated neuroimaging data.

    Tests spatial correspondence between parcellated maps while
    preserving spatial autocorrelation by rotating data on the
    cortical surface sphere.

    The primary interface is :meth:`spin_data`, which produces null
    maps via spherical rotations, and :meth:`compute_spin_pvalue`,
    which computes p-values from the spun maps. For convenience,
    :meth:`from_brainspace` provides a one-call wrapper around the
    BrainSpace toolbox.

    Parameters
    ----------
    random_state : int or None, optional
        Random seed for reproducibility. Default is ``None``.

    Examples
    --------
    >>> import numpy as np
    >>> spin = SpinTest(random_state=42)
    >>> data = np.random.randn(200)
    >>> parcellation = {"sphere_lh": lh_sphere, "sphere_rh": rh_sphere,
    ...                 "parcels_lh": parc_lh, "parcels_rh": parc_rh}
    >>> spun = spin.spin_data(data, parcellation, n_rotations=100)
    >>> p_val = spin.compute_spin_pvalue(data, spun)
    """

    def __init__(self, random_state: Optional[int] = None) -> None:
        self.random_state = random_state
        self._rng = np.random.default_rng(random_state)

    def spin_data(
        self,
        data: NDArray[np.floating],
        parcellation: dict,
        n_rotations: int = 1000,
    ) -> NDArray[np.floating]:
        """Perform spin rotation on parcellated data.

        Maps parcellated data to the fsaverage5 surface, applies
        independent spherical rotations to left and right hemispheres,
        then averages rotated vertex values back into parcels.

        Parameters
        ----------
        data : ndarray of shape (n_parcels,)
            Parcellated data vector. For Schaefer parcellations, the
            first half should contain left-hemisphere parcels and the
            second half right-hemisphere parcels.
        parcellation : dict
            Dictionary containing parcellation and surface information.
            Required keys depend on the method used:

            - ``"sphere_lh"``: Left hemisphere sphere surface dict with
              ``"vertices"`` and ``"faces"`` keys.
            - ``"sphere_rh"``: Right hemisphere sphere surface dict.
            - ``"parcels_lh"``: Left hemisphere parcel labels per vertex.
            - ``"parcels_rh"``: Right hemisphere parcel labels per vertex.
            - ``"n_parcels_lh"`` (optional): Number of LH parcels.
              Defaults to half of ``len(data)``.
        n_rotations : int, optional
            Number of random rotations. Default is 1000.

        Returns
        -------
        spun_data : ndarray of shape (n_rotations, n_parcels)
            Spun (rotated) data vectors, one per rotation.

        Raises
        ------
        ImportError
            If BrainSpace is not installed.
        ValueError
            If required keys are missing from ``parcellation``.
        """
        try:
            from brainspace.spin import permutation_test
        except ImportError:
            raise ImportError(
                "BrainSpace is required for spin_data. "
                "Install with: pip install brainspace"
            )

        data = np.asarray(data, dtype=np.float64)
        n_parcels = len(data)

        # Determine hemisphere split
        n_parcels_lh = parcellation.get("n_parcels_lh", n_parcels // 2)
        n_parcels_rh = n_parcels - n_parcels_lh

        # Build surface structures for BrainSpace
        sphere_lh = parcellation.get("sphere_lh")
        sphere_rh = parcellation.get("sphere_rh")
        if sphere_lh is None or sphere_rh is None:
            raise ValueError(
                "parcellation must contain 'sphere_lh' and 'sphere_rh' keys "
                "with 'vertices' and 'faces' arrays."
            )

        parcels_lh = parcellation.get("parcels_lh")
        parcels_rh = parcellation.get("parcels_rh")

        # Split data by hemisphere
        data_lh = data[:n_parcels_lh]
        data_rh = data[n_parcels_lh:]

        # Project parcel data to vertex space using sparse mapping
        if parcels_lh is not None and parcels_rh is not None:
            vertex_data_lh = self._parcel_to_vertex(data_lh, parcels_lh)
            vertex_data_rh = self._parcel_to_vertex(data_rh, parcels_rh)
        else:
            # If no parcel labels provided, treat data as vertex-level
            vertex_data_lh = data_lh
            vertex_data_rh = data_rh

        # Build BrainSpace surface objects
        surface_lh = self._make_surface(sphere_lh)
        surface_rh = self._make_surface(sphere_rh)

        # Generate random rotations
        seed = self._rng.integers(0, 2**31)
        result = permutation_test(
            [vertex_data_lh, vertex_data_rh],
            [surface_lh, surface_rh],
            n_rotations=n_rotations,
            random_state=seed,
        )

        # result is a list of two arrays (LH, RH), each (n_rotations, n_vertices)
        spun_lh = np.asarray(result[0])  # (n_rotations, n_vertices_lh)
        spun_rh = np.asarray(result[1])  # (n_rotations, n_vertices_rh)

        # Map back to parcel space
        if parcels_lh is not None and parcels_rh is not None:
            W_lh = self._build_parcel_weights(parcels_lh, n_parcels_lh)
            W_rh = self._build_parcel_weights(parcels_rh, n_parcels_rh)
            spun_parcels_lh = spun_lh @ W_lh.T  # (n_rotations, n_parcels_lh)
            spun_parcels_rh = spun_rh @ W_rh.T  # (n_rotations, n_parcels_rh)
        else:
            spun_parcels_lh = spun_lh
            spun_parcels_rh = spun_rh

        spun_data = np.concatenate([spun_parcels_lh, spun_parcels_rh], axis=1)
        return spun_data

    @staticmethod
    def compute_spin_pvalue(
        empirical: NDArray[np.floating],
        spun: NDArray[np.floating],
    ) -> NDArray[np.floating]:
        """Compute p-values from spin permutation results.

        For each parcel, the p-value is the proportion of spun
        permutations whose absolute value exceeds the absolute
        empirical value.

        Parameters
        ----------
        empirical : ndarray of shape (n_parcels,)
            Observed data values.
        spun : ndarray of shape (n_rotations, n_parcels)
            Spun (null) data from :meth:`spin_data`.

        Returns
        -------
        p_values : ndarray of shape (n_parcels,)
            Spin-test p-values for each parcel.
        """
        empirical = np.asarray(empirical, dtype=np.float64)
        spun = np.asarray(spun, dtype=np.float64)
        n_rotations = spun.shape[0]

        abs_empirical = np.abs(empirical)
        abs_spun = np.abs(spun)

        exceedances = np.sum(abs_spun >= abs_empirical[np.newaxis, :], axis=0)
        p_values = (exceedances + 1) / (n_rotations + 1)

        return p_values

    @staticmethod
    def from_brainspace(
        data: NDArray[np.floating],
        parcellation: dict,
        n_rotations: int = 1000,
        random_state: Optional[int] = None,
    ) -> NDArray[np.floating]:
        """Convenience wrapper: spin data using BrainSpace directly.

        Equivalent to creating a ``SpinTest`` instance and calling
        :meth:`spin_data`, but as a single static call.

        Parameters
        ----------
        data : ndarray of shape (n_parcels,)
            Parcellated data vector.
        parcellation : dict
            Parcellation specification (see :meth:`spin_data`).
        n_rotations : int, optional
            Number of rotations. Default is 1000.
        random_state : int or None, optional
            Random seed. Default is ``None``.

        Returns
        -------
        spun_data : ndarray of shape (n_rotations, n_parcels)
            Spun data vectors.
        """
        spin = SpinTest(random_state=random_state)
        return spin.spin_data(data, parcellation, n_rotations=n_rotations)

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _parcel_to_vertex(
        parcel_data: NDArray[np.floating],
        parcel_labels: NDArray[np.intp],
    ) -> NDArray[np.floating]:
        """Map parcel-level data to vertex-level using label assignments.

        Parameters
        ----------
        parcel_data : ndarray of shape (n_parcels,)
            Data values per parcel.
        parcel_labels : ndarray of shape (n_vertices,)
            Parcel label for each vertex.

        Returns
        -------
        vertex_data : ndarray of shape (n_vertices,)
            Data values per vertex.
        """
        # parcel_labels: 0 = medial wall, 1..n_parcels = parcels
        vertex_data = np.zeros(len(parcel_labels), dtype=np.float64)
        for p_idx in range(len(parcel_data)):
            mask = parcel_labels == (p_idx + 1)
            vertex_data[mask] = parcel_data[p_idx]
        return vertex_data

    @staticmethod
    def _build_parcel_weights(
        parcel_labels: NDArray[np.intp],
        n_parcels: int,
    ) -> NDArray[np.floating]:
        """Build sparse vertex-to-parcel averaging weight matrix.

        Parameters
        ----------
        parcel_labels : ndarray of shape (n_vertices,)
            Parcel label for each vertex (1-indexed, 0 = medial wall).
        n_parcels : int
            Number of parcels.

        Returns
        -------
        W : ndarray of shape (n_parcels, n_vertices)
            Weight matrix; ``W @ vertex_data`` yields parcel averages.
        """
        n_vertices = len(parcel_labels)
        W = np.zeros((n_parcels, n_vertices), dtype=np.float64)
        for p_idx in range(n_parcels):
            mask = parcel_labels == (p_idx + 1)
            count = np.sum(mask)
            if count > 0:
                W[p_idx, mask] = 1.0 / count
        return W

    @staticmethod
    def _make_surface(sphere_dict: dict) -> object:
        """Create a BrainSpace surface object from a dict.

        Parameters
        ----------
        sphere_dict : dict
            Must contain ``"vertices"`` (n_vertices x 3) and
            ``"faces"`` (n_faces x 3).

        Returns
        -------
        surface : brainspace Surface instance
        """
        try:
            from brainspace.mesh.mesh_elements import Surface
            return Surface(
                points=np.asarray(sphere_dict["vertices"]),
                cells=np.asarray(sphere_dict["faces"]),
            )
        except ImportError:
            raise ImportError(
                "BrainSpace is required. Install with: pip install brainspace"
            )
