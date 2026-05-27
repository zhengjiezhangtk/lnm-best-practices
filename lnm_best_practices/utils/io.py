"""File I/O utility functions for loading and saving neuroimaging data.

This module provides convenience wrappers around nibabel, scipy, and numpy
for common file operations used in lesion-network mapping workflows.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Tuple, Union

import numpy as np


PathLike = Union[str, Path]


def load_nifti(path: PathLike) -> Tuple[np.ndarray, np.ndarray]:
    """Load a NIfTI file and return the data array and affine matrix.

    Parameters
    ----------
    path : str or Path
        Path to a NIfTI file (.nii or .nii.gz).

    Returns
    -------
    data : np.ndarray
        Image data as a numpy array.
    affine : np.ndarray
        4x4 affine transformation matrix mapping voxel indices to world
        coordinates.

    Raises
    ------
    FileNotFoundError
        If the file does not exist.
    nibabel.filebasedimages.ImageFileError
        If the file is not a valid NIfTI image.

    Examples
    --------
    >>> data, affine = load_nifti("lesion_mask.nii.gz")
    >>> data.shape
    (91, 109, 91)
    """
    import nibabel as nib

    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"NIfTI file not found: {path}")

    img = nib.load(str(path))
    data = np.asarray(img.dataobj)
    affine = img.affine
    return data, affine


def save_nifti(
    data: np.ndarray,
    affine: np.ndarray,
    path: PathLike,
) -> None:
    """Save a numpy array as a NIfTI file.

    Parameters
    ----------
    data : np.ndarray
        Image data to save.
    affine : np.ndarray
        4x4 affine transformation matrix.
    path : str or Path
        Output path for the NIfTI file (.nii or .nii.gz).

    Raises
    ------
    ValueError
        If affine is not a 4x4 matrix.

    Examples
    --------
    >>> import numpy as np
    >>> data = np.zeros((91, 109, 91), dtype=np.float64)
    >>> affine = np.eye(4)
    >>> save_nifti(data, affine, "output.nii.gz")
    """
    import nibabel as nib

    path = Path(path)
    affine = np.asarray(affine)

    if affine.shape != (4, 4):
        raise ValueError(
            f"Affine matrix must be 4x4, got shape {affine.shape}"
        )

    img = nib.Nifti1Image(data.astype(np.float64), affine)
    nib.save(img, str(path))


def load_matrix(
    path: PathLike,
    format: str = "auto",
) -> np.ndarray:
    """Load a matrix from various file formats.

    Supports MATLAB (.mat), CSV (.csv), NumPy (.npy), and NumPy compressed
    (.npz) formats. When format is 'auto', the format is inferred from the
    file extension.

    Parameters
    ----------
    path : str or Path
        Path to the matrix file.
    format : str, optional
        File format: 'mat', 'csv', 'npy', or 'auto' (default: 'auto').
        When 'auto', the format is inferred from the file extension.

    Returns
    -------
    np.ndarray
        The loaded matrix as a 2-D numpy array.

    Raises
    ------
    FileNotFoundError
        If the file does not exist.
    ValueError
        If the format cannot be inferred or is not supported.

    Examples
    --------
    >>> matrix = load_matrix("connectome.mat")
    >>> matrix.shape
    (100, 100)
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Matrix file not found: {path}")

    if format == "auto":
        suffix = path.suffix.lower()
        format_map = {
            ".mat": "mat",
            ".csv": "csv",
            ".npy": "npy",
            ".npz": "npy",
        }
        if suffix not in format_map:
            raise ValueError(
                f"Cannot infer format from extension '{suffix}'. "
                f"Supported extensions: {list(format_map.keys())}"
            )
        format = format_map[suffix]

    if format == "mat":
        import scipy.io as sio

        mat = sio.loadmat(str(path))
        data_keys = [k for k in mat if not k.startswith("__")]
        if not data_keys:
            raise ValueError(f"No data variables found in .mat file: {path}")
        return np.asarray(mat[data_keys[0]])

    elif format == "csv":
        return np.loadtxt(str(path), delimiter=",")

    elif format == "npy":
        if path.suffix.lower() == ".npz":
            npz = np.load(str(path))
            keys = list(npz.keys())
            if not keys:
                raise ValueError(f"No arrays found in .npz file: {path}")
            return npz[keys[0]]
        return np.load(str(path))

    else:
        raise ValueError(f"Unsupported format: '{format}'. Use 'mat', 'csv', or 'npy'.")


def save_matrix(
    data: np.ndarray,
    path: PathLike,
    format: str = "auto",
) -> None:
    """Save a matrix to a file.

    Supports MATLAB (.mat), CSV (.csv), and NumPy (.npy) formats. When
    format is 'auto', the format is inferred from the file extension.

    Parameters
    ----------
    data : np.ndarray
        Matrix to save.
    path : str or Path
        Output file path.
    format : str, optional
        File format: 'mat', 'csv', 'npy', or 'auto' (default: 'auto').

    Raises
    ------
    ValueError
        If the format cannot be inferred or is not supported.

    Examples
    --------
    >>> import numpy as np
    >>> matrix = np.random.rand(100, 100)
    >>> save_matrix(matrix, "connectome.npy")
    """
    path = Path(path)
    data = np.asarray(data)

    if format == "auto":
        suffix = path.suffix.lower()
        format_map = {
            ".mat": "mat",
            ".csv": "csv",
            ".npy": "npy",
        }
        if suffix not in format_map:
            raise ValueError(
                f"Cannot infer format from extension '{suffix}'. "
                f"Supported extensions: {list(format_map.keys())}"
            )
        format = format_map[suffix]

    if format == "mat":
        import scipy.io as sio

        sio.savemat(str(path), {"data": data})

    elif format == "csv":
        np.savetxt(str(path), data, delimiter=",")

    elif format == "npy":
        np.save(str(path), data)

    else:
        raise ValueError(f"Unsupported format: '{format}'. Use 'mat', 'csv', or 'npy'.")


def load_atlas(path: PathLike) -> Tuple[np.ndarray, int]:
    """Load an atlas file and return label array and number of regions.

    Supports NIfTI atlas files. The returned label array is the raw voxel-wise
    atlas data, and region count is the number of unique non-zero labels.

    Parameters
    ----------
    path : str or Path
        Path to the atlas NIfTI file (.nii or .nii.gz).

    Returns
    -------
    labels : np.ndarray
        Voxel-wise integer label array.
    n_regions : int
        Number of unique non-zero regions in the atlas.

    Raises
    ------
    FileNotFoundError
        If the file does not exist.

    Examples
    --------
    >>> labels, n_regions = load_atlas("schaefer_100.nii.gz")
    >>> n_regions
    100
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Atlas file not found: {path}")

    data, _ = load_nifti(path)
    labels = np.round(data).astype(np.int32)
    n_regions = int(len(np.unique(labels[labels > 0])))
    return labels, n_regions


def load_symptoms(path: PathLike) -> np.ndarray:
    """Load a symptom scores file (CSV or TSV).

    The file is expected to contain numeric data with one subject per row.
    The first row is treated as a header and skipped. Delimiter is
    auto-detected from the file extension (.csv -> comma, .tsv -> tab).

    Parameters
    ----------
    path : str or Path
        Path to the symptom scores file (.csv or .tsv).

    Returns
    -------
    np.ndarray
        Symptom scores as a 2-D numpy array of shape (n_subjects, n_scores)
        or 1-D if only one score column exists.

    Raises
    ------
    FileNotFoundError
        If the file does not exist.
    ValueError
        If the file extension is not .csv or .tsv.

    Examples
    --------
    >>> symptoms = load_symptoms("nihss_scores.csv")
    >>> symptoms.shape
    (50, 5)
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Symptom file not found: {path}")

    suffix = path.suffix.lower()
    if suffix == ".csv":
        delimiter = ","
    elif suffix == ".tsv":
        delimiter = "\t"
    else:
        raise ValueError(
            f"Unsupported file extension '{suffix}'. Use .csv or .tsv."
        )

    data = np.loadtxt(str(path), delimiter=delimiter, skiprows=1)

    # Squeeze to 1-D if only one column
    if data.ndim == 2 and data.shape[1] == 1:
        data = data.ravel()

    return data
