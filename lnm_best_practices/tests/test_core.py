"""
Tests for core LNM modules.

Validates LNM computation, connectome loading, and lesion mapping.
"""

import numpy as np
import pytest
from pathlib import Path

from lnm_best_practices.core.lnm import LNM, LNMResult
from lnm_best_practices.core.connectome import Connectome


class TestLNM:
    """Tests for LNM computation."""

    @pytest.fixture
    def sample_data(self):
        """Create sample lesion matrix and connectome."""
        np.random.seed(42)
        n_subjects = 10
        n_parcels = 50

        # Random lesion matrix (normalized rows)
        M = np.random.rand(n_subjects, n_parcels)
        M = M / M.sum(axis=1, keepdims=True)

        # Random symmetric connectome
        C = np.random.rand(n_parcels, n_parcels)
        C = (C + C.T) / 2
        np.fill_diagonal(C, 0)

        # Random symptoms
        symptoms = np.random.randn(n_subjects)

        return M, C, symptoms

    def test_compute_lnm(self, sample_data):
        """Test standard LNM computation."""
        M, C, _ = sample_data
        lnm = LNM(M, C)
        result = lnm.compute()

        assert isinstance(result, LNMResult)
        assert result.lnm_map.shape == (M.shape[1],)
        assert result.method == 'lnm'
        assert np.all(np.isfinite(result.lnm_map))

    def test_compute_slnm(self, sample_data):
        """Test symptom-based LNM computation."""
        M, C, symptoms = sample_data
        lnm = LNM(M, C)
        result = lnm.compute_slnm(symptoms)

        assert isinstance(result, LNMResult)
        assert result.lnm_map.shape == (M.shape[1],)
        assert result.method == 'slnm'

    def test_compute_single(self, sample_data):
        """Test single lesion LNM computation."""
        M, C, _ = sample_data
        lnm = LNM(M, C)
        single_map = lnm.compute_single(0)

        assert single_map.shape == (M.shape[1],)
        assert np.all(np.isfinite(single_map))

    def test_normalize_map(self, sample_data):
        """Test LNM map normalization."""
        M, C, _ = sample_data
        lnm = LNM(M, C)
        result = lnm.compute()

        # Z-score normalization
        normalized = LNM.normalize_map(result.lnm_map, method='zscore')
        assert abs(np.mean(normalized)) < 1e-10
        assert abs(np.std(normalized) - 1.0) < 1e-10

    def test_dimension_mismatch(self):
        """Test that dimension mismatch raises error."""
        M = np.random.rand(10, 50)
        C = np.random.rand(30, 30)

        with pytest.raises(ValueError):
            LNM(M, C)

    def test_lnm_formula(self, sample_data):
        """Test that LNM follows the formula: LNM = sum(M @ C)."""
        M, C, _ = sample_data

        # Manual computation
        expected = np.sum(M @ C, axis=0)

        # LNM class
        lnm = LNM(M, C)
        result = lnm.compute()

        np.testing.assert_allclose(result.lnm_map, expected, rtol=1e-10)


class TestConnectome:
    """Tests for Connectome class."""

    def test_create_from_matrix(self):
        """Test creating Connectome from matrix."""
        C = np.random.rand(50, 50)
        C = (C + C.T) / 2

        conn = Connectome(n_rois=50)
        # Test degree map computation
        degree = conn.get_degree_map(C)
        assert degree.shape == (50,)

    def test_fisher_z_transform(self):
        """Test Fisher z transformation."""
        C = np.random.rand(10, 10)
        C = (C + C.T) / 2
        C = np.clip(C, -0.99, 0.99)

        z_matrix = Connectome.fisher_z_transform(C)
        assert z_matrix.shape == (10, 10)
        assert np.all(np.isfinite(z_matrix))

    def test_degree_map(self):
        """Test degree map computation."""
        C = np.array([
            [0, 1, 2],
            [1, 0, 3],
            [2, 3, 0]
        ], dtype=float)

        degree = Connectome.get_degree_map(C)

        expected = np.array([3, 4, 5])
        np.testing.assert_allclose(degree, expected)


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
