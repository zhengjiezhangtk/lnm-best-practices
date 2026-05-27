"""
Neuroimaging Analysis Module
----------------------------
Utility functions for symptom-based Lesion Network Mapping (sLNM) method validation analysis.

Functions
---------
voxel_outcome_correlation : correlate each voxel with outcome scores
pearson_rows              : correlate each row of X with a reference vector
nifti_getdata             : load and mask a NIfTI file into a flat brain array
recon_tmap                : visualize a parcel-wise map on the Schaefer fsaverage surface
"""

import numpy as np
import nibabel as nib
import matplotlib.pyplot as plt
import os
from matplotlib.colors import ListedColormap
from nilearn.plotting import plot_surf_stat_map


# ── Core correlation functions ────────────────────────────────────────────────

def voxel_outcome_correlation(brain_data, outcomes):
    """
    Compute Pearson correlation between each voxel and outcome scores.

    Parameters
    ----------
    brain_data : ndarray, shape (n_subjects, n_voxels)
        Brain connectivity data for each subject
    outcomes : ndarray, shape (n_subjects,) or (n_subjects, n_permutations)
        Outcome scores; can be multiple columns for permutation testing

    Returns
    -------
    correlations : ndarray, shape (n_voxels,) if single outcome, else (n_permutations, n_voxels)
    """
    outcomes = np.atleast_2d(outcomes) if outcomes.ndim == 1 else outcomes
    single = outcomes.shape[1] == 1

    brain_centered    = (brain_data - np.mean(brain_data, axis=0, keepdims=True)).astype(np.float32)
    outcomes_centered = (outcomes - np.mean(outcomes, axis=0, keepdims=True)).astype(np.float32)

    covariances  = outcomes_centered.T @ brain_centered
    var_outcomes = np.sum(outcomes_centered ** 2, axis=0)
    var_brain    = np.sum(brain_centered ** 2, axis=0)

    denominator  = np.sqrt(var_outcomes[:, None] * var_brain[None, :])
    denominator  = np.maximum(denominator, 1e-15)

    result = np.clip(covariances / denominator, -1, 1)

    return result.flatten() if single else result


def pearson_rows(X, y):
    """
    Compute Pearson correlation between each row of X and vector y.

    Handles NaN values by computing correlations only on valid pairs.

    Parameters
    ----------
    X : ndarray, shape (n_subjects, n_voxels)
        Array where each row is a pattern to correlate with y
    y : ndarray, shape (n_voxels,)
        Reference pattern to correlate against

    Returns
    -------
    correlations : ndarray, shape (n_subjects,)
        Pearson correlation coefficient for each row of X with y
    """
    valid_mask = ~(np.isnan(X) | np.isnan(y))

    X_mean = np.nansum(X * valid_mask, axis=1, keepdims=True) / np.sum(valid_mask, axis=1, keepdims=True)
    y_mean = np.nanmean(y)

    X_centered = np.where(valid_mask, X - X_mean, 0)
    y_centered = np.where(~np.isnan(y), y - y_mean, 0)

    numerator = np.sum(X_centered * y_centered, axis=1)
    n_valid   = np.sum(valid_mask, axis=1)
    X_std     = np.sqrt(np.sum(X_centered ** 2, axis=1) / n_valid)
    y_std     = np.nanstd(y)

    return numerator / (X_std * y_std * n_valid)


# ── NIfTI utilities ───────────────────────────────────────────────────────────

def nifti_getdata(nifti_path, brain_template_path=None):
    """
    Load a NIfTI file and return masked, flattened brain voxels.

    NaN values are interpolated using neighboring voxel values.

    Parameters
    ----------
    nifti_path          : str, path to NIfTI file
    brain_template_path : str, path to brain mask NIfTI (optional, uses default if None)

    Returns
    -------
    nifti_flattened : ndarray, shape (n_brain_voxels,)
    """
    if brain_template_path is None:
        brain_template_path = os.path.join(
            os.path.dirname(__file__),
            '..', 'data', 'templates', 'Taylor_NHB_MNI152_T1_2mm_brain_mask_dil.nii.gz'
        )

    brain_mask      = nib.load(brain_template_path).get_fdata() > 0
    nifti_flattened = nib.load(nifti_path).get_fdata()[brain_mask].flatten()

    # Interpolate NaN values using neighboring voxels
    nan_mask = np.isnan(nifti_flattened)
    if np.any(nan_mask):
        for i in np.where(nan_mask)[0]:
            neighbors = []
            if i > 0 and not np.isnan(nifti_flattened[i - 1]):
                neighbors.append(nifti_flattened[i - 1])
            if i < len(nifti_flattened) - 1 and not np.isnan(nifti_flattened[i + 1]):
                neighbors.append(nifti_flattened[i + 1])
            if neighbors:
                nifti_flattened[i] = np.mean(neighbors)

    return nifti_flattened


# ── Surface visualization ─────────────────────────────────────────────────────

class SchaeferVisualizer:
    """Visualize parcel-wise statistical maps on the Schaefer fsaverage5 surface."""

    def __init__(self, schaefer_fsaverage_dir=None, n_parcels=1000):
        if schaefer_fsaverage_dir is None:
            schaefer_fsaverage_dir = os.path.join(
                os.path.dirname(__file__),
                '..', 'data', 'templates', 'surf'
            )
        self.schaefer_dir = schaefer_fsaverage_dir
        self.n_parcels    = n_parcels

        self._load_surfaces()
        self._load_annotations()
        self._setup_colormap()

    def _load_surfaces(self):
        self.lh_mesh = nib.freesurfer.read_geometry(f'{self.schaefer_dir}/lh.inflated')
        self.rh_mesh = nib.freesurfer.read_geometry(f'{self.schaefer_dir}/rh.inflated')

    def _load_annotations(self):
        self.lh_annot = nib.freesurfer.read_annot(
            f'{self.schaefer_dir}/lh.Schaefer2018_{self.n_parcels}Parcels_7Networks_order.annot'
        )
        self.rh_annot = nib.freesurfer.read_annot(
            f'{self.schaefer_dir}/rh.Schaefer2018_{self.n_parcels}Parcels_7Networks_order.annot'
        )

    def _setup_colormap(self):
        colors = np.vstack([
            plt.cm.winter(np.linspace(1, 0, 128)),
            plt.cm.YlOrRd_r(np.linspace(0, 1, 128))
        ])
        self.hot_cold = ListedColormap(colors)

    def plot_t_map(self, t_map, title='', hemi='both', view='lateral', figsize=(10, 6)):
        """
        Plot a parcel-wise statistical map on the cortical surface.

        Parameters
        ----------
        t_map   : ndarray, shape (n_parcels,)
        title   : str, plot title
        hemi    : 'left', 'right', or 'both'
        view    : 'lateral', 'medial', etc.
        figsize : tuple

        Returns
        -------
        fig or (fig_lh, fig_rh)
        """
        n_per_hemi = self.n_parcels // 2
        figs = []

        for side, annot, mesh, idx_offset in [
            ('left',  self.lh_annot, self.lh_mesh, 0),
            ('right', self.rh_annot, self.rh_mesh, n_per_hemi)
        ]:
            if hemi not in [side, 'both']:
                continue

            surf = np.zeros_like(annot[0], dtype=float)
            hemi_map = t_map[idx_offset: idx_offset + n_per_hemi]

            for parcel_id in range(1, n_per_hemi + 1):
                surf[annot[0] == parcel_id] = t_map[idx_offset + parcel_id - 1]

            fig = plot_surf_stat_map(
                stat_map=surf,
                surf_mesh=mesh,
                hemi=side,
                cmap=self.hot_cold,
                colorbar=True,
                cbar_tick_format='%6.2f',
                darkness=None,
                vmax=np.max(np.abs(hemi_map)),
                title=f"{title} - {side.capitalize()} Hemisphere" if hemi == 'both' else title,
                view=view,
                figure=plt.figure(figsize=figsize)
            )
            figs.append(fig)

        return tuple(figs) if len(figs) > 1 else figs[0]


_global_visualizer = None

def recon_tmap(t_map, title='', hemi='left', view='lateral', schaefer_dir=None, n_parcels=None):
    """
    Visualize a parcel-wise map on the Schaefer fsaverage surface.

    Reuses a cached SchaeferVisualizer instance for efficiency.

    Parameters
    ----------
    t_map       : ndarray, shape (n_parcels,)
    title       : str
    hemi        : 'left', 'right', or 'both'
    view        : 'lateral', 'medial', etc.
    schaefer_dir: str, path to fsaverage5 Schaefer directory (optional)
    n_parcels   : int, number of parcels (inferred from t_map if None)
    """
    global _global_visualizer

    if n_parcels is None:
        n_parcels = len(t_map)

    if _global_visualizer is None or _global_visualizer.n_parcels != n_parcels:
        _global_visualizer = SchaeferVisualizer(schaefer_dir, n_parcels)

    return _global_visualizer.plot_t_map(t_map, title, hemi, view)


__all__ = [
    'voxel_outcome_correlation',
    'pearson_rows',
    'nifti_getdata',
    'recon_tmap',
    'SchaeferVisualizer',
]