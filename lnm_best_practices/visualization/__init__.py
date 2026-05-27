"""
Visualization Module
--------------------
Brain surface rendering, statistical charts, and NIfTI export utilities
for Lesion Network Mapping (LNM) analysis.

Submodules
----------
surface : cortical surface rendering using nilearn
plots   : statistical charts (null distributions, correlations, p-value maps)
export  : NIfTI / CIFTI file export utilities
"""

from .surface import BrainSurfacePlotter
from .plots import (
    plot_null_distribution,
    plot_correlation,
    plot_pvalue_map,
    plot_specificity_results,
)
from .export import (
    save_lnm_as_nifti,
    save_thresholded_map,
    save_as_cifti,
)

__all__ = [
    "BrainSurfacePlotter",
    "plot_null_distribution",
    "plot_correlation",
    "plot_pvalue_map",
    "plot_specificity_results",
    "save_lnm_as_nifti",
    "save_thresholded_map",
    "save_as_cifti",
]
