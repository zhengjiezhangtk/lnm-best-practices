"""
Surface Visualization
---------------------
Brain surface rendering utilities for Lesion Network Mapping (LNM) results.

This module provides the ``BrainSurfacePlotter`` class for visualizing
parcel-wise or vertex-wise statistical maps on inflated cortical surfaces
using nilearn's surface plotting functions.

Dependencies
------------
- nilearn >= 0.10
- nibabel
- matplotlib
- numpy
"""

from __future__ import annotations

import os
from typing import Optional, Sequence, Tuple, Union

import matplotlib.pyplot as plt
import nibabel as nib
import numpy as np
from matplotlib.colors import Colormap, ListedColormap
from nilearn.plotting import plot_surf_stat_map, plot_surf_roi


# ---------------------------------------------------------------------------
# Default paths – adjust to your project layout
# ---------------------------------------------------------------------------

_DEFAULT_SURF_DIR = os.path.join(
    os.path.dirname(__file__), "..", "data", "templates", "surf"
)


# ---------------------------------------------------------------------------
# Helper: build a symmetric hot-cold colormap
# ---------------------------------------------------------------------------

def _make_hot_cold_cmap(n: int = 256) -> ListedColormap:
    """Create a symmetric blue-white-red diverging colormap.

    Parameters
    ----------
    n : int
        Total number of colour entries (must be even).

    Returns
    -------
    ListedColormap
    """
    half = n // 2
    cold = plt.cm.Blues_r(np.linspace(0.2, 1, half))
    hot = plt.cm.Reds(np.linspace(0.2, 1, half))
    white = np.ones((1, 4))
    colors = np.vstack([cold, white, hot])
    return ListedColormap(colors)


# ===========================================================================
# BrainSurfacePlotter
# ===========================================================================

class BrainSurfacePlotter:
    """Render parcel-wise or vertex-wise data on the cortical surface.

    The class wraps nilearn's ``plot_surf_stat_map`` and ``plot_surf_roi``
    to provide a convenient interface for LNM visualisation.

    Parameters
    ----------
    surf_dir : str or None
        Directory containing FreeSurfer ``fsaverage5`` surface files.
        If ``None``, the default project path is used.
    n_parcels : int
        Number of Schaefer parcels (e.g. 100, 200, 1000).
    """

    def __init__(
        self,
        surf_dir: Optional[str] = None,
        n_parcels: int = 1000,
    ) -> None:
        self.surf_dir = surf_dir or _DEFAULT_SURF_DIR
        self.n_parcels = n_parcels
        self._load_surfaces()

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _load_surfaces(self) -> None:
        """Load FreeSurfer inflated meshes for both hemispheres."""
        lh_path = os.path.join(self.surf_dir, "lh.inflated")
        rh_path = os.path.join(self.surf_dir, "rh.inflated")

        if os.path.isfile(lh_path):
            self.lh_mesh = nib.freesurfer.read_geometry(lh_path)
        else:
            self.lh_mesh = None

        if os.path.isfile(rh_path):
            self.rh_mesh = nib.freesurfer.read_geometry(rh_path)
        else:
            self.rh_mesh = None

    @staticmethod
    def _resolve_mesh(
        hemi: str,
        lh_mesh,
        rh_mesh,
    ):
        """Return the mesh tuple for the requested hemisphere."""
        if hemi == "left":
            if lh_mesh is None:
                raise FileNotFoundError("Left hemisphere mesh not loaded.")
            return lh_mesh, "left"
        elif hemi == "right":
            if rh_mesh is None:
                raise FileNotFoundError("Right hemisphere mesh not loaded.")
            return rh_mesh, "right"
        else:
            # 'both' handled by caller
            raise ValueError("Use 'left' or 'right'; 'both' is handled upstream.")

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def plot_surface(
        self,
        data: np.ndarray,
        hemi: str = "both",
        view: str = "lateral",
        cmap: Union[str, Colormap] = "hot",
        threshold: float = 1.0,
        title: str = "",
        figsize: Tuple[int, int] = (12, 5),
        vmin: Optional[float] = None,
        vmax: Optional[float] = None,
        colorbar: bool = True,
    ) -> Union[plt.Figure, Tuple[plt.Figure, plt.Figure]]:
        """Plot statistical data on the cortical surface.

        Parameters
        ----------
        data : np.ndarray
            Vertex-wise or parcel-wise values to display.
        hemi : str
            ``'left'``, ``'right'``, or ``'both'``.
        view : str
            View angle: ``'lateral'``, ``'medial'``, ``'dorsal'``, etc.
        cmap : str or Colormap
            Matplotlib colormap name or object.
        threshold : float
            Threshold below which values are not displayed.
        title : str
            Figure title.
        figsize : tuple of int
            Figure size ``(width, height)`` in inches.
        vmin, vmax : float or None
            Colour-map range.  Computed from *data* if ``None``.
        colorbar : bool
            Whether to draw a colour-bar.

        Returns
        -------
        matplotlib.figure.Figure or tuple of Figure
            A single figure when *hemi* is ``'left'`` or ``'right'``,
            or a ``(left_fig, right_fig)`` tuple when *hemi* is ``'both'``.
        """
        if vmin is None:
            vmin = float(np.nanmin(data))
        if vmax is None:
            vmax = float(np.nanmax(data))

        hemispheres: list[str] = (
            ["left", "right"] if hemi == "both" else [hemi]
        )

        figs: list[plt.Figure] = []
        for h in hemispheres:
            mesh, _ = self._resolve_mesh(h, self.lh_mesh, self.rh_mesh)
            fig_i = plt.figure(figsize=figsize)
            plot_surf_stat_map(
                surf_mesh=mesh,
                stat_map=data,
                hemi=h,
                view=view,
                cmap=cmap,
                threshold=threshold,
                vmin=vmin,
                vmax=vmax,
                colorbar=colorbar,
                title=title if hemi != "both" else f"{title} ({h})",
                figure=fig_i,
                darkness=0.6,
            )
            figs.append(fig_i)

        return tuple(figs) if len(figs) == 2 else figs[0]

    def plot_lnm_map(
        self,
        lnm_map: np.ndarray,
        atlas: Optional[str] = None,
        title: str = "LNM Map",
        hemi: str = "both",
        view: str = "lateral",
        cmap: Union[str, Colormap] = "hot",
        threshold: float = 1.0,
        figsize: Tuple[int, int] = (12, 5),
    ) -> Union[plt.Figure, Tuple[plt.Figure, plt.Figure]]:
        """Plot a Lesion Network Mapping result on the cortical surface.

        This is a convenience wrapper around :meth:`plot_surface` with
        LNM-specific defaults.

        Parameters
        ----------
        lnm_map : np.ndarray
            Parcel-wise LNM connectivity values (length must match
            ``n_parcels`` or vertex count).
        atlas : str or None
            Path to the atlas NIfTI used for parcellation (informational).
        title : str
            Figure title.
        hemi : str
            ``'left'``, ``'right'``, or ``'both'``.
        view : str
            View angle.
        cmap : str or Colormap
            Colormap.
        threshold : float
            Display threshold.
        figsize : tuple of int
            Figure size.

        Returns
        -------
        Figure or tuple of Figure
        """
        if atlas is not None:
            title = f"{title}\natlas: {os.path.basename(atlas)}"
        return self.plot_surface(
            data=lnm_map,
            hemi=hemi,
            view=view,
            cmap=cmap,
            threshold=threshold,
            title=title,
            figsize=figsize,
        )

    def plot_comparison(
        self,
        map1: np.ndarray,
        map2: np.ndarray,
        titles: Tuple[str, str] = ("Map 1", "Map 2"),
        hemi: str = "left",
        view: str = "lateral",
        cmap: Union[str, Colormap] = "hot",
        threshold: float = 1.0,
        figsize: Tuple[int, int] = (16, 5),
    ) -> plt.Figure:
        """Side-by-side comparison of two maps on the cortical surface.

        Parameters
        ----------
        map1, map2 : np.ndarray
            Parcel-wise or vertex-wise arrays of the same shape.
        titles : tuple of str
            Titles for the left and right panels.
        hemi : str
            Hemisphere to plot (each panel shows the same hemisphere).
        view : str
            View angle.
        cmap : str or Colormap
            Colormap.
        threshold : float
            Display threshold.
        figsize : tuple of int
            Figure size.

        Returns
        -------
        matplotlib.figure.Figure
            A figure with two side-by-side surface panels.
        """
        mesh, _ = self._resolve_mesh(hemi, self.lh_mesh, self.rh_mesh)

        fig, axes = plt.subplots(1, 2, figsize=figsize,
                                  subplot_kw={"projection": "3d"})

        for ax, data, ttl in zip(axes, [map1, map2], titles):
            plot_surf_stat_map(
                surf_mesh=mesh,
                stat_map=data,
                hemi=hemi,
                view=view,
                cmap=cmap,
                threshold=threshold,
                colorbar=True,
                title=ttl,
                axes=ax,
                darkness=0.6,
            )

        fig.tight_layout()
        return fig
