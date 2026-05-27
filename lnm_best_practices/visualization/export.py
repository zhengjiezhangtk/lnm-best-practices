"""
NIfTI / CIFTI Export
--------------------
Utilities for saving Lesion Network Mapping (LNM) results as neuroimaging
files that can be loaded into standard viewers (e.g. FSLeyes, Connectome
Workbench).

Dependencies
------------
- nibabel >= 3.2
- numpy
- pathlib (stdlib)
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Optional, Union

import nibabel as nib
import numpy as np


# ---------------------------------------------------------------------------
# NIfTI export
# ---------------------------------------------------------------------------

def save_lnm_as_nifti(
    lnm_map: np.ndarray,
    atlas_path: Union[str, Path],
    output_path: Union[str, Path],
    description: str = "LNM connectivity map",
) -> Path:
    """Map parcel-wise LNM values back to MNI voxels and save as NIfTI.

    Each voxel in the atlas is assigned the value of its corresponding
    parcel in *lnm_map*.

    Parameters
    ----------
    lnm_map : np.ndarray
        1-D array of length ``n_parcels`` containing the LNM value for
        each parcel.
    atlas_path : str or Path
        Path to the parcellation atlas NIfTI (integer labels).
    output_path : str or Path
        Destination ``.nii`` or ``.nii.gz`` file.
    description : str
        NIfTI header description field.

    Returns
    -------
    pathlib.Path
        Resolved path of the saved file.

    Raises
    ------
    FileNotFoundError
        If *atlas_path* does not exist.
    ValueError
        If the atlas contains parcel IDs not covered by *lnm_map*.
    """
    atlas_path = Path(atlas_path)
    output_path = Path(output_path)

    if not atlas_path.is_file():
        raise FileNotFoundError(f"Atlas not found: {atlas_path}")

    atlas_img = nib.load(atlas_path)
    atlas_data = atlas_img.get_fdata().astype(int)

    # Determine unique parcel IDs (exclude 0 = background)
    unique_ids = np.unique(atlas_data)
    unique_ids = unique_ids[unique_ids > 0]

    if unique_ids.max() >= len(lnm_map):
        raise ValueError(
            f"Atlas contains parcel ID {unique_ids.max()} but lnm_map "
            f"has only {len(lnm_map)} entries."
        )

    # Build the volumetric map
    vol = np.zeros(atlas_data.shape, dtype=np.float32)
    for pid in unique_ids:
        vol[atlas_data == pid] = lnm_map[pid]

    out_img = nib.Nifti1Image(vol, affine=atlas_img.affine,
                               header=atlas_img.header)
    out_img.set_data_dtype(np.float32)

    # Update description
    hdr = out_img.header
    if hasattr(hdr, "set_description"):
        hdr.set_description(description)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    nib.save(out_img, str(output_path))
    return output_path.resolve()


def save_thresholded_map(
    lnm_map: np.ndarray,
    p_values: np.ndarray,
    atlas_path: Union[str, Path],
    output_path: Union[str, Path],
    alpha: float = 0.05,
    description: str = "Thresholded LNM map",
) -> Path:
    """Save a p-value thresholded LNM map as NIfTI.

    Only parcels whose p-value is strictly less than *alpha* retain their
    LNM value; all other voxels are set to zero.

    Parameters
    ----------
    lnm_map : np.ndarray
        1-D array of parcel-wise LNM values.
    p_values : np.ndarray
        1-D array of p-values (same length as *lnm_map*).
    atlas_path : str or Path
        Path to the parcellation atlas NIfTI.
    output_path : str or Path
        Destination file path.
    alpha : float
        Significance threshold.
    description : str
        NIfTI header description.

    Returns
    -------
    pathlib.Path
        Resolved path of the saved file.
    """
    lnm_map = np.asarray(lnm_map, dtype=np.float32)
    p_values = np.asarray(p_values, dtype=float)

    # Zero out non-significant parcels
    thresholded = lnm_map.copy()
    thresholded[p_values >= alpha] = 0.0

    return save_lnm_as_nifti(
        lnm_map=thresholded,
        atlas_path=atlas_path,
        output_path=output_path,
        description=description,
    )


# ---------------------------------------------------------------------------
# CIFTI export (optional)
# ---------------------------------------------------------------------------

def save_as_cifti(
    lnm_map: np.ndarray,
    output_path: Union[str, Path],
    n_parcels: int = 1000,
    parcellation: str = "Schaefer400",
) -> Path:
    """Save an LNM map as a CIFTI-2 dense scalar file (.dscalar.nii).

    This function creates a simple CIFTI file with a single row (map).
    It requires that the BrainordinatesAxis can be constructed from the
    parcel count.  For full CIFTI support with surface geometry, a
    ``cifti`` library (e.g. ``cifti-tools`` or ``nibabel >= 4.0``) is
    needed.

    Parameters
    ----------
    lnm_map : np.ndarray
        1-D array of parcel-wise values.
    output_path : str or Path
        Destination ``.dscalar.nii`` file.
    n_parcels : int
        Number of parcels (default 1000).
    parcellation : str
        Name label for the parcellation scheme.

    Returns
    -------
    pathlib.Path
        Resolved path of the saved file.

    Raises
    ------
    ImportError
        If nibabel does not support CIFTI-2 writing.
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    try:
        # nibabel >= 4.0 has CIFTI-2 support
        from nibabel.cifti2 import (
            Cifti2Image,
            Cifti2Header,
            Cifti2Matrix,
            Cifti2MatrixIndicesMap,
            Cifti2Parcel,
            Cifti2ParcelsAxis,
            Cifti2ScalarAxis,
        )

        brain_model = Cifti2ParcelsAxis(
            name=[f"parcel_{i}" for i in range(n_parcels)],
            voxel_indices_ijk=[],
            surface_indices=[],
        )
        scalar_axis = Cifti2ScalarAxis(name=["LNM connectivity"])
        matrix = Cifti2Matrix()
        matrix.append(scalar_axis)
        matrix.append(brain_model)
        header = Cifti2Header(matrix)

        data = lnm_map.reshape(1, -1).astype(np.float32)
        img = Cifti2Image(data, header=header)
        nib.save(img, str(output_path))

    except ImportError:
        raise ImportError(
            "CIFTI-2 writing requires nibabel >= 4.0 with CIFTI support. "
            "Install a recent nibabel or use save_lnm_as_nifti instead."
        )

    return output_path.resolve()
