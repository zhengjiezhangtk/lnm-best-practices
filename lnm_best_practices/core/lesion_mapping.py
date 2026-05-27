"""
Lesion-to-atlas mapping module.

Maps voxel-level lesion masks to parcellation space for LNM computation.

References:
    lesionnetworkmapping/utils/map_voxel2schaeferMelbourne.m
"""

import numpy as np
import nibabel as nib
from pathlib import Path
from typing import List, Optional, Union


class LesionMapper:
    """Map lesion masks to atlas space.

    Parameters
    ----------
    atlas_path : str or Path
        Path to atlas NIfTI file
    n_parcels : int, optional
        Number of parcels in atlas (auto-detected if not provided)

    Examples
    --------
    >>> mapper = LesionMapper(atlas_path='schaefer1000.nii.gz')
    >>> M = mapper.compute_lesion_matrix(lesion_paths)
    """

    def __init__(
        self,
        atlas_path: Union[str, Path],
        n_parcels: Optional[int] = None,
    ):
        self.atlas_path = Path(atlas_path)
        self.atlas_data, self.atlas_affine = self._load_atlas(self.atlas_path)
        self.n_parcels = n_parcels or int(np.max(self.atlas_data))

    @staticmethod
    def _load_atlas(path: Path) -> tuple:
        """Load atlas NIfTI file.

        Returns
        -------
        tuple
            (atlas_data, affine_matrix)
        """
        img = nib.load(str(path))
        return img.get_fdata().astype(int), img.affine

    def map_single_lesion(
        self,
        lesion_path: Union[str, Path],
        normalize: bool = True,
    ) -> np.ndarray:
        """Map a single lesion mask to atlas space.

        Parameters
        ----------
        lesion_path : str or Path
            Path to lesion NIfTI file
        normalize : bool, default=True
            Normalize so values sum to 1

        Returns
        -------
        np.ndarray
            Lesion vector in atlas space (n_parcels,)
        """
        lesion_img = nib.load(str(lesion_path))
        lesion_data = lesion_img.get_fdata()

        # Ensure same space
        if lesion_data.shape != self.atlas_data.shape:
            raise ValueError(
                f"Lesion shape {lesion_data.shape} != atlas shape {self.atlas_data.shape}"
            )

        # Map to atlas using accumarray-like operation
        lesion_vector = np.zeros(self.n_parcels)
        lesion_mask = lesion_data > 0

        for parcel_id in range(1, self.n_parcels + 1):
            parcel_mask = self.atlas_data == parcel_id
            overlap = np.sum(lesion_mask & parcel_mask)
            if overlap > 0:
                lesion_vector[parcel_id - 1] = overlap

        if normalize and np.sum(lesion_vector) > 0:
            lesion_vector = lesion_vector / np.sum(lesion_vector)

        return lesion_vector

    def compute_lesion_matrix(
        self,
        lesion_paths: List[Union[str, Path]],
        normalize: bool = True,
    ) -> np.ndarray:
        """Compute lesion matrix for multiple subjects.

        Parameters
        ----------
        lesion_paths : list of str or Path
            Paths to lesion NIfTI files
        normalize : bool, default=True
            Normalize each row to sum to 1

        Returns
        -------
        np.ndarray
            Lesion matrix (n_subjects x n_parcels)
        """
        n_subjects = len(lesion_paths)
        M = np.zeros((n_subjects, self.n_parcels))

        for i, path in enumerate(lesion_paths):
            M[i] = self.map_single_lesion(path, normalize=normalize)

        return M

    @staticmethod
    def load_lesion_matrix(path: Union[str, Path]) -> np.ndarray:
        """Load pre-computed lesion matrix.

        Parameters
        ----------
        path : str or Path
            Path to lesion matrix file (.csv, .npy, .mat)

        Returns
        -------
        np.ndarray
            Lesion matrix (n_subjects x n_parcels)
        """
        path = Path(path)
        suffix = path.suffix.lower()

        if suffix == '.npy':
            return np.load(path)
        elif suffix == '.csv':
            return np.loadtxt(path, delimiter=',')
        elif suffix in ('.mat',):
            from scipy.io import loadmat
            data = loadmat(path)
            for key, val in data.items():
                if isinstance(val, np.ndarray) and val.ndim == 2:
                    return val.astype(np.float64)
            raise ValueError(f"No 2D array found in {path}")
        else:
            raise ValueError(f"Unsupported format: {suffix}")
