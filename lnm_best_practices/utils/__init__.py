"""Utility modules for LNM best practices.

Includes:
    - I/O functions (NIfTI, matrix loading)
    - Input validation
    - neuroimage_analysis: Reused from symptom_lnm repo (Treeratana et al.)
"""

from .io import (
    load_nifti,
    save_nifti,
    load_matrix,
    save_matrix,
    load_atlas,
    load_symptoms,
)
from .validation import (
    validate_lesion_matrix,
    validate_connectome,
    validate_symptoms,
    validate_atlas,
    check_dimensions_match,
)
from .neuroimage_analysis import (
    voxel_outcome_correlation,
    pearson_rows,
    nifti_getdata,
    recon_tmap,
    SchaeferVisualizer,
)

__all__ = [
    "load_nifti",
    "save_nifti",
    "load_matrix",
    "save_matrix",
    "load_atlas",
    "load_symptoms",
    "validate_lesion_matrix",
    "validate_connectome",
    "validate_symptoms",
    "validate_atlas",
    "check_dimensions_match",
    "voxel_outcome_correlation",
    "pearson_rows",
    "nifti_getdata",
    "recon_tmap",
    "SchaeferVisualizer",
]
