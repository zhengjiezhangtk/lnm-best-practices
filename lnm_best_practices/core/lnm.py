"""Core Lesion Network Mapping (LNM) computation module.

Implements the linear form of Lesion Network Mapping:
    - Standard LNM:  LNM = sum(M * C)
    - Symptom LNM:   sLNM = sv' * (M * C)

Where:
    M : lesion matrix (n_patients, n_parcels), each row sums to 1
    C : normative connectome (n_parcels, n_parcels)
    sv: symptom vector (n_patients,), z-scored

This implementation follows the algorithms from:
    - lnm_nulls/lnm_compute.m (Zalesky et al.)
    - lnm_nulls/clinical_lesions/compute_lnm.m
    - lesionnetworkmapping/Example1_LNM_step123.m
    - lesionnetworkmapping/Example4_sLNM_step123.m
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, Optional

import numpy as np
from numpy.typing import NDArray


@dataclass
class LNMResult:
    """Container for LNM computation results.

    Attributes:
        lnm_map: The computed LNM map of shape (n_parcels,).
        method: The computation method used ('lnm', 'slnm', or 'single').
        symptom_vector: The symptom vector used (only for sLNM).
    """

    lnm_map: NDArray[np.floating]
    method: Literal["lnm", "slnm", "single"]
    symptom_vector: Optional[NDArray[np.floating]] = None


class LNM:
    """Lesion Network Mapping computation class.

    Provides a high-level interface for computing standard LNM,
    symptom-based LNM (sLNM), and single-subject LNM maps.

    Algorithm follows lnm_compute.m:
        LNM = ind_lesion' * w / K
    where ind_lesion is the lesion assignment vector, w is the connectome,
    and K is the number of lesions.

    Parameters
    ----------
    lesion_matrix : ndarray of shape (n_subjects, n_parcels)
        Binary or weighted lesion matrix.
    connectome : ndarray of shape (n_parcels, n_parcels)
        Normative functional connectivity matrix.

    Examples
    --------
    >>> import numpy as np
    >>> M = np.random.rand(10, 50)
    >>> C = np.random.rand(50, 50)
    >>> C = (C + C.T) / 2
    >>> lnm = LNM(M, C)
    >>> result = lnm.compute()
    """

    def __init__(
        self,
        lesion_matrix: NDArray[np.floating],
        connectome: NDArray[np.floating],
    ) -> None:
        self.lesion_matrix = np.asarray(lesion_matrix, dtype=np.float64)
        self.connectome = np.asarray(connectome, dtype=np.float64)

        if self.lesion_matrix.ndim != 2:
            raise ValueError(
                f"lesion_matrix must be 2-D, got shape {self.lesion_matrix.shape}"
            )
        if self.connectome.ndim != 2 or self.connectome.shape[0] != self.connectome.shape[1]:
            raise ValueError(
                f"connectome must be a square 2-D array, got shape {self.connectome.shape}"
            )
        if self.lesion_matrix.shape[1] != self.connectome.shape[0]:
            raise ValueError(
                f"Dimension mismatch: lesion_matrix has {self.lesion_matrix.shape[1]} "
                f"columns but connectome has {self.connectome.shape[0]} rows"
            )

    def compute(self) -> LNMResult:
        """Compute group-level LNM map.

        Follows lnm_compute.m:
            lnm = ind_lesion' * w / K

        Returns
        -------
        LNMResult
            Result object containing the LNM map and method info.
        """
        lnm_map = compute_lnm(self.lesion_matrix, self.connectome)
        return LNMResult(lnm_map=lnm_map, method="lnm")

    def compute_slnm(self, symptoms: NDArray[np.floating]) -> LNMResult:
        """Compute symptom-based LNM map.

        Follows Example4_sLNM_step123.m:
            sLNM = sv' * (M * C)

        Parameters
        ----------
        symptoms : ndarray of shape (n_subjects,)
            Clinical symptom scores.

        Returns
        -------
        LNMResult
            Result object containing the sLNM map.
        """
        symptoms = np.asarray(symptoms, dtype=np.float64).ravel()
        slnm_map = compute_slnm(self.lesion_matrix, self.connectome, symptoms)
        return LNMResult(lnm_map=slnm_map, method="slnm", symptom_vector=symptoms)

    def compute_single(self, subject_idx: int) -> NDArray[np.floating]:
        """Compute LNM map for a single subject.

        Follows clinical_lesions/compute_lnm.m:
            lnm = x_norm * C

        Parameters
        ----------
        subject_idx : int
            Index of the subject in the lesion matrix.

        Returns
        -------
        ndarray of shape (n_parcels,)
            Single-subject LNM map.
        """
        lesion_row = self.lesion_matrix[subject_idx]
        return compute_single_lnm(lesion_row, self.connectome)

    def compute_distributional(
        self,
        n_trials: int = 1000,
        seed: Optional[int] = None,
    ) -> NDArray[np.floating]:
        """Compute distributional LNM (sampling one region per lesion).

        Follows clinical_lesions/compute_lnm.m:
            For each trial, randomly sample one region per lesion,
            then compute mean connectivity.

        Parameters
        ----------
        n_trials : int
            Number of random sampling trials.
        seed : int, optional
            Random seed.

        Returns
        -------
        ndarray of shape (n_trials, n_parcels)
            Distributional LNM maps.
        """
        return compute_distributional_lnm(
            self.lesion_matrix, self.connectome, n_trials, seed
        )

    @staticmethod
    def normalize_map(
        lnm_map: NDArray[np.floating],
        method: str = "zscore",
    ) -> NDArray[np.floating]:
        """Normalize an LNM map.

        Parameters
        ----------
        lnm_map : ndarray of shape (n_parcels,)
            Raw LNM map.
        method : str, optional
            Normalization method ('zscore' or 'minmax'). Default is 'zscore'.

        Returns
        -------
        ndarray of shape (n_parcels,)
            Normalized map.
        """
        return normalize_lnm(lnm_map, method=method)


def compute_lnm(
    lesion_matrix: NDArray[np.floating],
    connectome: NDArray[np.floating],
) -> NDArray[np.floating]:
    """Compute group-level LNM map using the formula LNM = sum(M * C).

    Follows lnm_compute.m line 31: lnm = ind_lesion' * w / K

    Parameters
    ----------
    lesion_matrix : numpy.ndarray, shape (n_patients, n_parcels)
        Binary (or weighted) lesion matrix.
    connectome : numpy.ndarray, shape (n_parcels, n_parcels)
        Normative functional (or structural) connectivity matrix.

    Returns
    -------
    lnm_map : numpy.ndarray, shape (n_parcels,)
        Group-level LNM sensitivity map.
    """
    lesion_matrix = np.asarray(lesion_matrix, dtype=np.float64)
    connectome = np.asarray(connectome, dtype=np.float64)

    if lesion_matrix.ndim != 2:
        raise ValueError(f"lesion_matrix must be 2-D, got shape {lesion_matrix.shape}")
    if connectome.ndim != 2 or connectome.shape[0] != connectome.shape[1]:
        raise ValueError(f"connectome must be a square 2-D array, got shape {connectome.shape}")
    if lesion_matrix.shape[1] != connectome.shape[0]:
        raise ValueError(
            f"Dimension mismatch: lesion_matrix has {lesion_matrix.shape[1]} "
            f"columns but connectome has {connectome.shape[0]} rows"
        )

    # Normalize each lesion row so that it sums to 1 (probability mask)
    # Follows compute_lnm.m: x_norm = x ./ repmat(Nregs, 1, length(C))
    row_sums = lesion_matrix.sum(axis=1, keepdims=True)
    row_sums = np.where(row_sums == 0, 1.0, row_sums)
    M = lesion_matrix / row_sums

    # LNM = sum(M * C) -> (n_parcels,)
    # Follows lnm_compute.m: lnm = ind_lesion' * w / K
    lnm_map = np.nansum(M @ connectome, axis=0)

    return lnm_map


def compute_slnm(
    lesion_matrix: NDArray[np.floating],
    connectome: NDArray[np.floating],
    symptoms: NDArray[np.floating],
) -> NDArray[np.floating]:
    """Compute group-level symptom LNM map: sLNM = sv' * (M * C).

    Follows Example4_sLNM_step123.m.

    Parameters
    ----------
    lesion_matrix : numpy.ndarray, shape (n_patients, n_parcels)
        Binary (or weighted) lesion matrix.
    connectome : numpy.ndarray, shape (n_parcels, n_parcels)
        Normative connectivity matrix.
    symptoms : numpy.ndarray, shape (n_patients,)
        Clinical symptom scores.

    Returns
    -------
    slnm_map : numpy.ndarray, shape (n_parcels,)
        Symptom LNM sensitivity map.
    """
    lesion_matrix = np.asarray(lesion_matrix, dtype=np.float64)
    connectome = np.asarray(connectome, dtype=np.float64)
    symptoms = np.asarray(symptoms, dtype=np.float64).ravel()

    if lesion_matrix.ndim != 2:
        raise ValueError(f"lesion_matrix must be 2-D, got shape {lesion_matrix.shape}")
    if connectome.ndim != 2 or connectome.shape[0] != connectome.shape[1]:
        raise ValueError(f"connectome must be a square 2-D array, got shape {connectome.shape}")
    if lesion_matrix.shape[1] != connectome.shape[0]:
        raise ValueError(
            f"Dimension mismatch: lesion_matrix has {lesion_matrix.shape[1]} "
            f"columns but connectome has {connectome.shape[0]} rows"
        )
    if symptoms.shape[0] != lesion_matrix.shape[0]:
        raise ValueError(
            f"symptoms length ({symptoms.shape[0]}) does not match "
            f"number of patients ({lesion_matrix.shape[0]})"
        )

    # Normalize each lesion row
    row_sums = lesion_matrix.sum(axis=1, keepdims=True)
    row_sums = np.where(row_sums == 0, 1.0, row_sums)
    M = lesion_matrix / row_sums

    # Z-score the symptom vector
    sv_std = np.nanstd(symptoms)
    if sv_std == 0:
        sv_z = np.zeros_like(symptoms)
    else:
        sv_z = (symptoms - np.nanmean(symptoms)) / sv_std

    # sLNM = sv' * (M * C)
    slnm_map = sv_z @ (M @ connectome)

    return slnm_map


def compute_single_lnm(
    lesion_row: NDArray[np.floating],
    connectome: NDArray[np.floating],
) -> NDArray[np.floating]:
    """Compute the LNM connectivity map for a single lesion.

    Follows clinical_lesions/compute_lnm.m: lnm = x_norm * C

    Parameters
    ----------
    lesion_row : numpy.ndarray, shape (n_parcels,)
        Single patient's lesion vector.
    connectome : numpy.ndarray, shape (n_parcels, n_parcels)
        Normative connectivity matrix.

    Returns
    -------
    lnm_single : numpy.ndarray, shape (n_parcels,)
        Per-parcel connectivity strength for this lesion.
    """
    lesion_row = np.asarray(lesion_row, dtype=np.float64).ravel()
    connectome = np.asarray(connectome, dtype=np.float64)

    if connectome.ndim != 2 or connectome.shape[0] != connectome.shape[1]:
        raise ValueError(f"connectome must be a square 2-D array, got shape {connectome.shape}")
    if lesion_row.shape[0] != connectome.shape[0]:
        raise ValueError(
            f"Dimension mismatch: lesion_row has {lesion_row.shape[0]} "
            f"elements but connectome has {connectome.shape[0]} rows"
        )

    # Normalize
    n_regions = np.sum(lesion_row > 0)
    if n_regions == 0:
        raise RuntimeError("lesion_row contains no non-zero parcels")
    x_norm = lesion_row / n_regions

    # LNM = x_norm * C
    lnm_single = x_norm @ connectome

    return lnm_single


def compute_distributional_lnm(
    lesion_matrix: NDArray[np.floating],
    connectome: NDArray[np.floating],
    n_trials: int = 1000,
    seed: Optional[int] = None,
) -> NDArray[np.floating]:
    """Compute distributional LNM by sampling one region per lesion.

    Follows clinical_lesions/compute_lnm.m lines 10-16:
        For each trial, randomly sample one region per lesion,
        then compute mean connectivity.

    Parameters
    ----------
    lesion_matrix : ndarray of shape (n_subjects, n_parcels)
        Binary lesion matrix.
    connectome : ndarray of shape (n_parcels, n_parcels)
        Connectivity matrix.
    n_trials : int
        Number of random sampling trials.
    seed : int, optional
        Random seed.

    Returns
    -------
    lnm_dist : ndarray of shape (n_trials, n_parcels)
        Distributional LNM maps.
    """
    lesion_matrix = np.asarray(lesion_matrix, dtype=np.float64)
    connectome = np.asarray(connectome, dtype=np.float64)
    rng = np.random.default_rng(seed)

    n_subjects, n_parcels = lesion_matrix.shape

    # Find lesion indices (ii, jj) like MATLAB's find(x)
    # ii = subject index, jj = parcel index
    ii, jj = np.where(lesion_matrix > 0)

    lnm_dist = np.zeros((n_trials, n_parcels), dtype=np.float64)

    for n in range(n_trials):
        # For each subject, randomly sample one lesioned parcel
        # Follows: ind_region = accumarray(ii, jj, [], @(x) x(randi(length(x))))
        selected = np.zeros(n_subjects, dtype=int)
        for subj in range(n_subjects):
            subj_parcels = jj[ii == subj]
            if len(subj_parcels) > 0:
                selected[subj] = rng.choice(subj_parcels)

        # Compute mean connectivity for selected parcels
        lnm_dist[n] = np.mean(connectome[selected], axis=0)

    return lnm_dist


def normalize_lnm(
    lnm_map: NDArray[np.floating],
    method: str = "zscore",
) -> NDArray[np.floating]:
    """Normalise an LNM map.

    Parameters
    ----------
    lnm_map : numpy.ndarray, shape (n_parcels,)
        Raw LNM or sLNM map.
    method : {'zscore', 'minmax'}, optional
        Normalisation method.

    Returns
    -------
    lnm_normalised : numpy.ndarray, shape (n_parcels,)
        Normalised map.
    """
    lnm_map = np.asarray(lnm_map, dtype=np.float64).ravel()
    method = method.lower()

    if method == "zscore":
        std = np.nanstd(lnm_map)
        if std == 0:
            return np.zeros_like(lnm_map)
        return (lnm_map - np.nanmean(lnm_map)) / std

    if method == "minmax":
        vmin = np.nanmin(lnm_map)
        vmax = np.nanmax(lnm_map)
        denom = vmax - vmin
        if denom == 0:
            return np.zeros_like(lnm_map)
        return (lnm_map - vmin) / denom

    raise ValueError(f"Unknown normalisation method '{method}'. Use 'zscore' or 'minmax'.")
