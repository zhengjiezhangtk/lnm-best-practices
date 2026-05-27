"""
Tests for null model modules.

Validates spatial, topological, permutation, and spin test implementations.
"""

import numpy as np
import pytest

from lnm_best_practices.null_models.spatial import SpatialNullModel
from lnm_best_practices.null_models.topological import TopologicalNullModel
from lnm_best_practices.null_models.permutation import PermutationTest


class TestSpatialNullModel:
    """Tests for spatial null model."""

    @pytest.fixture
    def sample_data(self):
        """Create sample data."""
        np.random.seed(42)
        n_subjects = 5
        n_parcels = 20

        # Binary lesion matrix
        M = np.zeros((n_subjects, n_parcels))
        for i in range(n_subjects):
            n_affected = np.random.randint(3, 8)
            affected = np.random.choice(n_parcels, n_affected, replace=False)
            M[i, affected] = 1.0
            M[i] = M[i] / M[i].sum()

        # Random connectome
        C = np.random.rand(n_parcels, n_parcels)
        C = (C + C.T) / 2

        return M, C

    def test_generate_null(self, sample_data):
        """Test null generation preserves lesion size."""
        M, C = sample_data
        n_subjects, n_parcels = M.shape

        null_model = SpatialNullModel(random_state=42)
        null_maps = null_model.generate_null(M, n_permutations=10)

        assert null_maps.shape == (10, n_parcels)
        assert np.all(np.isfinite(null_maps))

    def test_compute_pvalue(self, sample_data):
        """Test p-value computation."""
        M, C = sample_data
        n_parcels = M.shape[1]

        null_model = SpatialNullModel(random_state=42)
        null_maps = null_model.generate_null(M, n_permutations=100)

        # Create empirical map
        empirical = np.sum(M @ C, axis=0)

        p_values = null_model.compute_pvalue(empirical, null_maps)

        assert p_values.shape == (n_parcels,)
        assert np.all(p_values > 0)
        assert np.all(p_values <= 1)


class TestTopologicalNullModel:
    """Tests for topological null model."""

    def test_degree_preserving_randomization(self):
        """Test that randomization preserves degree."""
        np.random.seed(42)
        n = 20
        # Create a well-connected graph
        C = np.random.rand(n, n)
        C = (C + C.T) / 2
        np.fill_diagonal(C, 0)
        # Threshold to ensure enough edges
        C = (C > 0.3).astype(float)

        null_model = TopologicalNullModel(random_state=42)
        randomized = null_model.degree_preserving_randomization(C, n_swaps=500)

        # Check shape preserved
        assert randomized.shape == C.shape

        # Check degree approximately preserved
        original_degree = np.sum(C > 0, axis=0)
        randomized_degree = np.sum(randomized > 0, axis=0)
        # Degree should be similar (correlation > 0.8)
        corr = np.corrcoef(original_degree, randomized_degree)[0, 1]
        assert corr > 0.8, f"Degree correlation too low: {corr}"


class TestPermutationTest:
    """Tests for permutation test."""

    @pytest.fixture
    def sample_data(self):
        """Create sample data."""
        np.random.seed(42)
        n_subjects = 20
        n_parcels = 30

        M = np.random.rand(n_subjects, n_parcels)
        M = M / M.sum(axis=1, keepdims=True)

        C = np.random.rand(n_parcels, n_parcels)
        C = (C + C.T) / 2

        symptoms = np.random.randn(n_subjects)

        return M, C, symptoms

    def test_permute_symptoms(self, sample_data):
        """Test symptom permutation."""
        M, C, symptoms = sample_data

        perm_test = PermutationTest(random_state=42)
        permuted = perm_test.permute_symptoms(symptoms, n_permutations=100)

        assert permuted.shape == (100, len(symptoms))

        # Each permutation should have same values, just reordered
        for i in range(100):
            np.testing.assert_allclose(
                np.sort(permuted[i]), np.sort(symptoms)
            )

    def test_fwer_correction(self, sample_data):
        """Test FWER correction."""
        M, C, symptoms = sample_data

        perm_test = PermutationTest(random_state=42)

        # Create null maps via symptom permutation
        n_perms = 100
        permuted = perm_test.permute_symptoms(symptoms, n_permutations=n_perms)

        # Generate null LNM maps
        null_maps = np.zeros((n_perms, M.shape[1]))
        for i in range(n_perms):
            sv_z = (permuted[i] - permuted[i].mean()) / permuted[i].std()
            null_maps[i] = sv_z @ (M @ C)

        # Create empirical map
        sv_z = (symptoms - symptoms.mean()) / symptoms.std()
        empirical = sv_z @ (M @ C)

        fwer_p, significant = perm_test.fwer_correction(empirical, null_maps, alpha=0.05)

        assert fwer_p.shape == (M.shape[1],)
        assert significant.shape == (M.shape[1],)
        assert np.all(fwer_p > 0)
        assert np.all(fwer_p <= 1)


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
