"""Input validation utilities for lesion-network mapping workflows.

This module provides functions to validate the format, shape, and contents
of common data structures used in LNM analyses, including lesion matrices,
connectomes, symptom vectors, and atlas files.
"""

from __future__ import annotations

from pathlib import Path
from typing import Sequence, Union

import numpy as np


PathLike = Union[str, Path]


def validate_lesion_matrix(matrix: np.ndarray) -> np.ndarray:
    """Validate a lesion overlap matrix.

    Checks that the input is a 2-D numeric numpy array with finite values
    and that all entries are 0 or 1 (binary mask).

    Parameters
    ----------
    matrix : np.ndarray
        Lesion matrix of shape (n_subjects, n_voxels) or similar.

    Returns
    -------
    np.ndarray
        The validated matrix (as a float64 numpy array).

    Raises
    ------
    TypeError
        If the input is not array-like.
    ValueError
        If the array is not 2-D, contains non-finite values, or contains
        values outside {0, 1}.

    Examples
    --------
    >>> import numpy as np
    >>> mat = np.array([[1, 0, 1], [0, 1, 0]])
    >>> validate_lesion_matrix(mat)
    array([[1., 0., 1.],
           [0., 1., 0.]])
    """
    matrix = np.asarray(matrix, dtype=np.float64)

    if matrix.ndim != 2:
        raise ValueError(
            f"Lesion matrix must be 2-D, got {matrix.ndim}-D array "
            f"with shape {matrix.shape}"
        )

    if not np.all(np.isfinite(matrix)):
        raise ValueError("Lesion matrix contains non-finite values (NaN or Inf).")

    unique_vals = np.unique(matrix)
    if not np.all(np.isin(unique_vals, [0.0, 1.0])):
        raise ValueError(
            f"Lesion matrix must be binary (0s and 1s only). "
            f"Found unique values: {unique_vals}"
        )

    return matrix

def validate_connectome(matrix: np.ndarray) -> np.ndarray:
    """Validate a connectivity (connectome) matrix.

    Checks that the input is a 2-D square numeric numpy array with finite
    values, and that it is symmetric (within a small tolerance).

    Parameters
    ----------
    matrix : np.ndarray
        Connectivity matrix of shape (n_regions, n_regions).

    Returns
    -------
    np.ndarray
        The validated matrix (as a float64 numpy array).

    Raises
    ------
    TypeError
        If the input is not array-like.
    ValueError
        If the array is not 2-D, not square, contains non-finite values,
        or is not symmetric.

    Examples
    --------
    >>> import numpy as np
    >>> conn = np.array([[0.0, 0.5, 0.3], [0.5, 0.0, 0.7], [0.3, 0.7, 0.0]])
    >>> validate_connectome(conn)
    array([[0. , 0.5, 0.3],
           [0.5, 0. , 0.7],
           [0.3, 0.7, 0. ]])
    """
    matrix = np.asarray(matrix, dtype=np.float64)

    if matrix.ndim != 2:
        raise ValueError(
            f"Connectome must be 2-D, got {matrix.ndim}-D array "
            f"with shape {matrix.shape}"
        )

    if matrix.shape[0] != matrix.shape[1]:
        raise ValueError(
            f"Connectome must be square, got shape {matrix.shape}"
        )

    if not np.all(np.isfinite(matrix)):
        raise ValueError("Connectome contains non-finite values (NaN or Inf).")

    if not np.allclose(matrix, matrix.T, atol=1e-6):
        raise ValueError("Connectome matrix is not symmetric.")

    return matrix

def validate_symptoms(
    symptoms: np.ndarray,
    n_subjects: int,
) -> np.ndarray:
    """Validate a symptom scores vector or matrix.

    Checks that the input is a numeric numpy array whose leading dimension
    matches the expected number of subjects and that all values are finite.

    Parameters
    ----------
    symptoms : np.ndarray
        Symptom scores, either 1-D (n_subjects,) or 2-D
        (n_subjects, n_scores).
    n_subjects : int
        Expected number of subjects.

    Returns
    -------
    np.ndarray
        The validated symptom array (as a float64 numpy array).

    Raises
    ------
    ValueError
        If the array is not 1-D or 2-D, the leading dimension does not
        match n_subjects, or contains non-finite values.

    Examples
    --------
    >>> import numpy as np
    >>> scores = np.array([3.0, 5.0, 2.0, 7.0])
    >>> validate_symptoms(scores, n_subjects=4)
    array([3., 5., 2., 7.])
    """
    symptoms = np.asarray(symptoms, dtype=np.float64)

    if symptoms.ndim not in (1, 2):
        raise ValueError(
            f"Symptoms must be 1-D or 2-D, got {symptoms.ndim}-D array."
        )

    if symptoms.shape[0] != n_subjects:
        raise ValueError(
            f"Number of subjects mismatch: expected {n_subjects}, "
            f"got {symptoms.shape[0]} rows."
        )

    if not np.all(np.isfinite(symptoms)):
        raise ValueError("Symptom scores contain non-finite values (NaN or Inf).")

    return symptoms

def validate_atlas(atlas_path: PathLike) -> dict:
    """Validate an atlas NIfTI file.

    Loads the atlas and performs basic checks: the file must exist, be a
    valid NIfTI image, and contain integer-like labels.

    Parameters
    ----------
    atlas_path : str or Path
        Path to the atlas NIfTI file (.nii or .nii.gz).

    Returns
    -------
    dict
        Dictionary with atlas metadata:
        - "path" (str): path to the atlas file.
        - "shape" (tuple): spatial dimensions of the atlas.
        - "n_regions" (int): number of unique non-zero regions.
        - "labels" (np.ndarray): sorted unique non-zero label values.

    Raises
    ------
    FileNotFoundError
        If the file does not exist.
    ValueError
        If the atlas contains no non-zero labels.

    Examples
    --------
    >>> info = validate_atlas("schaefer_100.nii.gz")
    >>> info["n_regions"]
    100
    """
    from utils.io import load_atlas, load_nifti

    atlas_path = Path(atlas_path)
    if not atlas_path.exists():
        raise FileNotFoundError(f"Atlas file not found: {atlas_path}")

    data, _ = load_nifti(atlas_path)
    labels_array, n_regions = load_atlas(atlas_path)

    if n_regions == 0:
        raise ValueError(f"Atlas contains no non-zero regions: {atlas_path}")

    unique_labels = np.sort(np.unique(labels_array[labels_array > 0]))

    return {
        "path": str(atlas_path),
        "shape": data.shape,
        "n_regions": n_regions,
        "labels": unique_labels,
    }

def check_dimensions_match(*arrays: np.ndarray) -> bool:
    """Check that all provided arrays have the same shape.

    This is useful for verifying that lesion matrices, connectivity
    matrices, and other data structures are dimensionally compatible.

    Parameters
    ----------
    *arrays : np.ndarray
        Variable number of arrays to compare.

    Returns
    -------
    bool
        True if all arrays have the same shape.

    Raises
    ------
    ValueError
        If fewer than two arrays are provided, or if shapes do not match.

    Examples
    --------
    >>> import numpy as np
    >>> a = np.zeros((50, 100))
    >>> b = np.ones((50, 100))
    >>> check_dimensions_match(a, b)
    True
    """
    if len(arrays) < 2:
        raise ValueError(
            f"At least two arrays are required, got {len(arrays)}."
        )

    shapes = [np.asarray(a).shape for a in arrays]

    for i in range(1, len(shapes)):
        if shapes[i] != shapes[0]:
            raise ValueError(
                f"Dimension mismatch: array 0 has shape {shapes[0]}, "
                f"but array {i} has shape {shapes[i]}."
            )

    return True