"""
Connectome loading and preprocessing module for Lesion Network Mapping.

This module provides functionality to load, preprocess, and analyze functional
connectivity matrices used in Lesion Network Mapping (LNM). It supports the
Schaefer1000+Melbourne54 atlas parcellation (1054 ROIs total: 54 subcortical
Melbourne regions followed by 1000 cortical Schaefer regions).

Reference implementations:
    - lesionnetworkmapping (MATLAB): setup_project.m, Example1_LNM_step123.m
    - lnm_nulls (MATLAB): lnm_compute.m, dcsbm.m

Typical usage:
    >>> from lnm_best_practices.core.connectome import Connectome
    >>> conn = Connectome()
    >>> matrix = conn.load_connectome("/path/to/GSP1000_FC_Schaefer1000Melbourne54.mat")
    >>> matrix_z = conn.fisher_z_transform(matrix)
    >>> degree = conn.get_degree_map(matrix_z)
"""

from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Optional, Union

import numpy as np
from numpy.typing import NDArray

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

N_CORTICAL: int = 1000
"""Number of cortical ROIs (Schaefer 1000-parcel atlas)."""

N_SUBCORTICAL: int = 54
"""Number of subcortical ROIs (Melbourne atlas)."""

N_TOTAL: int = N_CORTICAL + N_SUBCORTICAL
"""Total number of ROIs (Schaefer1000 + Melbourne54 = 1054)."""

SUBCORTICAL_INDICES: tuple[int, int] = (0, N_SUBCORTICAL)
"""Python slice boundaries for subcortical ROIs (0-based: [0, 54))."""

CORTICAL_INDICES: tuple[int, int] = (N_SUBCORTICAL, N_TOTAL)
"""Python slice boundaries for cortical ROIs (0-based: [54, 1054))."""

# Default path to GSP1000 normative connectome (if available in the project)
_DEFAULT_CONNECTOME_DIR = Path(__file__).resolve().parent.parent.parent / \
    "code" / "lesionnetworkmapping" / "data" / "normative_connectome"
DEFAULT_GSP1000_PATH: Optional[Path] = (
    _DEFAULT_CONNECTOME_DIR / "GSP1000_FC_Schaefer1000Melbourne54.mat"
    if (_DEFAULT_CONNECTOME_DIR / "GSP1000_FC_Schaefer1000Melbourne54.mat").exists()
    else None
)
DEFAULT_GSP1000_Z_PATH: Optional[Path] = (
    _DEFAULT_CONNECTOME_DIR / "GSP1000_FCz_Schaefer1000Melbourne54.mat"
    if (_DEFAULT_CONNECTOME_DIR / "GSP1000_FCz_Schaefer1000Melbourne54.mat").exists()
    else None
)

GSP1000_DOWNLOAD_INFO: str = (
    "The GSP1000 normative connectome can be obtained from the Harvard Dataverse:\n"
    "  https://dataverse.harvard.edu/dataverse/cohenlab\n"
    "Download the dataset and place the .mat file at:\n"
    "  <project>/code/lesionnetworkmapping/data/normative_connectome/\n"
    "Expected files:\n"
    "  - GSP1000_FC_Schaefer1000Melbourne54.mat  (Fisher r values)\n"
    "  - GSP1000_FCz_Schaefer1000Melbourne54.mat (Fisher z-transformed values)"
)


class Connectome:
    """Load, preprocess, and analyze functional connectivity matrices.

    This class encapsulates all operations on connectivity matrices used in
    Lesion Network Mapping.  It supports the Schaefer1000+Melbourne54 parcellation
    with 1054 ROIs (54 subcortical + 1000 cortical).

    Attributes:
        n_rois: Total number of ROIs in the atlas (default 1054).
        n_subcortical: Number of subcortical ROIs (default 54).
        n_cortical: Number of cortical ROIs (default 1000).
        matrix: The most recently loaded / processed connectivity matrix.
    """

    def __init__(
        self,
        n_rois: int = N_TOTAL,
        n_subcortical: int = N_SUBCORTICAL,
    ) -> None:
        """Initialize the Connectome handler.

        Args:
            n_rois: Total number of ROIs expected in the connectivity matrix.
                    Default is 1054 (Schaefer1000 + Melbourne54).
            n_subcortical: Number of subcortical ROIs at the start of the matrix.
                           Default is 54 (Melbourne atlas).
        """
        self.n_rois: int = n_rois
        self.n_subcortical: int = n_subcortical
        self.n_cortical: int = n_rois - n_subcortical
        self.matrix: Optional[NDArray[np.float64]] = None

    # ------------------------------------------------------------------
    # Loading
    # ------------------------------------------------------------------

    def load_connectome(
        self,
        path: Union[str, Path],
        format: str = "auto",
        fisher_z: bool = False,
    ) -> NDArray[np.float64]:
        """Load a functional connectivity matrix from disk.

        Supports .mat (MATLAB), .csv, and .npy formats.  When *format* is
        ``'auto'`` the format is inferred from the file extension.

        For .mat files the loader looks for common variable names used in
        the LNM codebase (``FC``, ``connectome``, ``matrix``, ``corr``).
        If none match, the first 2-D array found in the file is returned.

        Args:
            path: Path to the connectivity matrix file.
            format: One of ``'auto'``, ``'mat'``, ``'csv'``, ``'npy'``.
            fisher_z: If ``True``, apply Fisher r-to-z transform to the
                      loaded matrix before returning.

        Returns:
            A 2-D numpy array of shape ``(n_rois, n_rois)`` representing
            the functional connectivity matrix.

        Raises:
            FileNotFoundError: If *path* does not exist.
            ValueError: If the loaded matrix shape is inconsistent with
                        the expected atlas dimensions, or if the format
                        is unsupported.
        """
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"Connectome file not found: {path}")

        fmt = format.lower()
        if fmt == "auto":
            suffix = path.suffix.lower()
            if suffix == ".mat":
                fmt = "mat"
            elif suffix == ".csv":
                fmt = "csv"
            elif suffix == ".npy":
                fmt = "npy"
            else:
                raise ValueError(
                    f"Cannot auto-detect format for extension '{suffix}'. "
                    "Please specify format='mat', 'csv', or 'npy'."
                )

        if fmt == "mat":
            matrix = self._load_mat(path)
        elif fmt == "csv":
            matrix = self._load_csv(path)
        elif fmt == "npy":
            matrix = self._load_npy(path)
        else:
            raise ValueError(
                f"Unsupported format '{fmt}'. Use 'mat', 'csv', or 'npy'."
            )

        # Validate dimensions
        if matrix.ndim != 2:
            raise ValueError(
                f"Expected a 2-D connectivity matrix, got {matrix.ndim}-D array "
                f"with shape {matrix.shape}."
            )

        if matrix.shape[0] != matrix.shape[1]:
            raise ValueError(
                f"Connectivity matrix must be square, got shape {matrix.shape}."
            )

        if matrix.shape[0] != self.n_rois:
            logger.warning(
                "Matrix has %d ROIs but expected %d. "
                "Proceeding with loaded dimensions.",
                matrix.shape[0],
                self.n_rois,
            )

        logger.info(
            "Loaded connectome from %s  shape=%s  dtype=%s",
            path,
            matrix.shape,
            matrix.dtype,
        )

        if fisher_z:
            matrix = self.fisher_z_transform(matrix)

        self.matrix = matrix
        return matrix

    # ------------------------------------------------------------------
    # Private loaders
    # ------------------------------------------------------------------

    @staticmethod
    def _load_mat(path: Path) -> NDArray[np.float64]:
        """Load a MATLAB .mat file containing a connectivity matrix.

        Searches for common variable names used in LNM workflows.  If none
        match, returns the first 2-D numeric array found.

        Args:
            path: Path to the .mat file.

        Returns:
            2-D numpy array.
        """
        from scipy.io import loadmat

        data = loadmat(str(path))
        # Common variable names in the LNM codebase
        candidate_keys = ["FC", "connectome", "matrix", "corr", "C", "w"]
        for key in candidate_keys:
            if key in data:
                arr = np.asarray(data[key], dtype=np.float64)
                if arr.ndim == 2:
                    logger.debug("Found matrix under key '%s' in %s", key, path)
                    return arr

        # Fallback: find the first 2-D numeric array (skip MATLAB metadata)
        for key, value in data.items():
            if key.startswith("__"):
                continue
            arr = np.asarray(value, dtype=np.float64)
            if arr.ndim == 2 and arr.shape[0] > 1 and arr.shape[1] > 1:
                logger.debug(
                    "Using fallback key '%s' (shape %s) from %s",
                    key,
                    arr.shape,
                    path,
                )
                return arr

        raise ValueError(
            f"No 2-D numeric array found in {path}. "
            f"Available keys: {list(data.keys())}"
        )

    @staticmethod
    def _load_csv(path: Path) -> NDArray[np.float64]:
        """Load a connectivity matrix from a CSV file.

        Args:
            path: Path to the .csv file.

        Returns:
            2-D numpy array.
        """
        try:
            matrix = np.loadtxt(str(path), delimiter=",", dtype=np.float64)
        except ValueError:
            # Some CSV files have headers
            matrix = np.genfromtxt(str(path), delimiter=",", dtype=np.float64)
            # Drop header row if it contains NaN
            if matrix.ndim == 2 and np.any(np.isnan(matrix[0, :])):
                matrix = matrix[1:, :]
        return matrix

    @staticmethod
    def _load_npy(path: Path) -> NDArray[np.float64]:
        """Load a connectivity matrix from a .npy file.

        Args:
            path: Path to the .npy file.

        Returns:
            2-D numpy array.
        """
        return np.load(str(path)).astype(np.float64)

    # ------------------------------------------------------------------
    # Preprocessing
    # ------------------------------------------------------------------

    @staticmethod
    def fisher_z_transform(matrix: NDArray[np.float64]) -> NDArray[np.float64]:
        """Apply Fisher r-to-z transformation to a connectivity matrix.

        Transforms Pearson correlation values (r) to z-scores using:

        .. math::

            z = 0.5 \\cdot \\ln\\!\\left(\\frac{1 + r}{1 - r}\\right)

        Values of exactly +1 or -1 are clipped to avoid infinity.  The
        diagonal is preserved (set to 0 after transform if it was 0).

        This mirrors the Fisher z-transformed connectome stored in
        ``GSP1000_FCz_Schaefer1000Melbourne54.mat`` in the LNM codebase.

        Args:
            matrix: Square connectivity matrix of Pearson *r* values in
                    the range [-1, 1].

        Returns:
            Fisher z-transformed connectivity matrix.

        Raises:
            ValueError: If values fall outside [-1, 1] after clipping.
        """
        diag = np.diag(matrix).copy()

        # Clip to avoid arctanh singularities at exactly +/-1
        clipped = np.clip(matrix, -1.0 + 1e-15, 1.0 - 1e-15)
        z_matrix = np.arctanh(clipped)  # arctanh(r) = 0.5 * ln((1+r)/(1-r))

        # Restore diagonal: keep original diagonal values (typically 0 or 1)
        np.fill_diagonal(z_matrix, diag)

        logger.debug("Applied Fisher r-to-z transform  min=%.4f  max=%.4f",
                      np.nanmin(z_matrix - np.diag(np.diag(z_matrix))),
                      np.nanmax(z_matrix - np.diag(np.diag(z_matrix))))
        return z_matrix

    @staticmethod
    def normalize(
        matrix: NDArray[np.float64],
        method: str = "zscore",
    ) -> NDArray[np.float64]:
        """Normalize a connectivity matrix.

        Supported methods:

        - ``'zscore'``: Z-score normalization of off-diagonal elements.
          Each element is transformed as ``(x - mean) / std``.
        - ``'minmax'``: Min-max scaling to [0, 1].
        - ``'rank'``: Rank-based normalization (converts values to ranks,
          then scales to [0, 1]).

        The diagonal is excluded from statistics and preserved as-is.

        Args:
            matrix: Square connectivity matrix.
            method: Normalization method. One of ``'zscore'``, ``'minmax'``,
                    ``'rank'``.

        Returns:
            Normalized connectivity matrix.

        Raises:
            ValueError: If *method* is not recognized.
        """
        method = method.lower()
        n = matrix.shape[0]
        result = matrix.copy()
        diag_vals = np.diag(matrix).copy()

        # Extract off-diagonal elements
        mask = ~np.eye(n, dtype=bool)
        off_diag = matrix[mask]

        if method == "zscore":
            mean_val = np.nanmean(off_diag)
            std_val = np.nanstd(off_diag)
            if std_val == 0:
                logger.warning("Std is 0 during z-score normalization; "
                               "returning zero matrix off-diagonal.")
                result[mask] = 0.0
            else:
                result[mask] = (off_diag - mean_val) / std_val

        elif method == "minmax":
            min_val = np.nanmin(off_diag)
            max_val = np.nanmax(off_diag)
            denom = max_val - min_val
            if denom == 0:
                logger.warning("Range is 0 during min-max normalization; "
                               "returning zero matrix off-diagonal.")
                result[mask] = 0.0
            else:
                result[mask] = (off_diag - min_val) / denom

        elif method == "rank":
            from scipy.stats import rankdata
            ranks = rankdata(off_diag, nan_policy="omit")
            result[mask] = (ranks - 1) / max(len(ranks) - 1, 1)

        else:
            raise ValueError(
                f"Unknown normalization method '{method}'. "
                "Choose from 'zscore', 'minmax', 'rank'."
            )

        np.fill_diagonal(result, diag_vals)
        return result

    # ------------------------------------------------------------------
    # Analysis
    # ------------------------------------------------------------------

    @staticmethod
    def get_degree_map(
        matrix: NDArray[np.float64],
        threshold: Optional[float] = None,
    ) -> NDArray[np.float64]:
        """Compute the node strength (weighted degree) map.

        For a weighted connectivity matrix, the node strength of node *i* is:

        .. math::

            s_i = \\sum_{j} w_{ij}

        This corresponds to ``s = sum(w) / N`` used in the MATLAB LNM codebase
        (see ``lnm_compute.m``), but here we return the raw sum rather than
        the mean, to be consistent with standard graph theory conventions.

        If *threshold* is provided, the matrix is binarized (values >= threshold
        become 1, others become 0) before computing degree.

        Args:
            matrix: Square connectivity matrix.
            threshold: Optional threshold for binarization.  If ``None``,
                       weighted degree (node strength) is returned.

        Returns:
            1-D array of length ``n_rois`` containing the degree / strength
            of each node.
        """
        if threshold is not None:
            binary = (matrix >= threshold).astype(np.float64)
            np.fill_diagonal(binary, 0.0)
            degree = np.sum(binary, axis=1)
        else:
            # Node strength: sum of absolute weights per row, excluding diagonal
            m = matrix.copy()
            np.fill_diagonal(m, 0.0)
            degree = np.sum(np.abs(m), axis=1)
        return degree

    def get_subcortical_degree(
        self,
        matrix: NDArray[np.float64],
        n_subcortical: Optional[int] = None,
    ) -> NDArray[np.float64]:
        """Get the degree / strength values for subcortical regions only.

        In the Schaefer1000+Melbourne54 atlas, subcortical regions occupy
        the first *n_subcortical* rows/columns (indices 0..53 by default).

        This is useful for analyses that focus on subcortical connectivity,
        such as examining lesion connectivity to basal ganglia or thalamic
        structures.

        Args:
            matrix: Square connectivity matrix (1054 x 1054).
            n_subcortical: Number of subcortical ROIs at the start of the
                           matrix.  Defaults to ``self.n_subcortical`` (54).

        Returns:
            1-D array of length *n_subcortical* with the degree / strength
            of each subcortical node.
        """
        if n_subcortical is None:
            n_subcortical = self.n_subcortical

        degree = self.get_degree_map(matrix)
        return degree[:n_subcortical]

    # ------------------------------------------------------------------
    # Convenience / factory
    # ------------------------------------------------------------------

    @classmethod
    def from_default_gsp1000(
        cls,
        fisher_z: bool = False,
        connectome_dir: Optional[Union[str, Path]] = None,
    ) -> "Connectome":
        """Create a Connectome pre-loaded with the GSP1000 normative connectome.

        Searches for ``GSP1000_FC_Schaefer1000Melbourne54.mat`` (or the
        Fisher-z variant) in the default project location or in the
        specified directory.

        Args:
            fisher_z: If ``True``, load the Fisher z-transformed variant
                      (``GSP1000_FCz_Schaefer1000Melbourne54.mat``).
            connectome_dir: Optional directory containing the .mat files.
                            If ``None``, uses the default project path.

        Returns:
            A ``Connectome`` instance with the matrix already loaded.

        Raises:
            FileNotFoundError: If the connectome file cannot be found.
        """
        if connectome_dir is not None:
            base = Path(connectome_dir)
        else:
            base = _DEFAULT_CONNECTOME_DIR

        if fisher_z:
            fname = "GSP1000_FCz_Schaefer1000Melbourne54.mat"
        else:
            fname = "GSP1000_FC_Schaefer1000Melbourne54.mat"

        fpath = base / fname
        if not fpath.exists():
            raise FileNotFoundError(
                f"GSP1000 connectome not found at {fpath}.\n"
                f"{GSP1000_DOWNLOAD_INFO}"
            )

        conn = cls()
        conn.load_connectome(fpath)
        return conn

    def __repr__(self) -> str:
        loaded = "loaded" if self.matrix is not None else "not loaded"
        shape = self.matrix.shape if self.matrix is not None else None
        return (
            f"Connectome(n_rois={self.n_rois}, "
            f"n_subcortical={self.n_subcortical}, "
            f"n_cortical={self.n_cortical}, "
            f"matrix={loaded}, shape={shape})"
        )
