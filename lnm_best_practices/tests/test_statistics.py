"""
Tests for statistics modules.

Validates statistical tests, correction methods, and specificity testing.
"""

import numpy as np
import pytest

from lnm_best_practices.statistics.tests import (
    one_sample_t_test,
    two_sample_t_test,
    pearson_correlation,
    glm_t_test,
)
from lnm_best_practices.statistics.correction import (
    bonferroni_correction,
    fdr_correction,
    fwer_correction,
)
from lnm_best_practices.statistics.specificity import SpecificityTest


class TestStatisticalTests:
    """Tests for statistical test functions."""

    def test_one_sample_t_test(self):
        """Test one-sample t-test."""
        np.random.seed(42)
        data = np.random.randn(100) + 0.5  # Mean = 0.5

        t_stat, p_value = one_sample_t_test(data, null_mean=0)

        assert isinstance(t_stat, float)
        assert isinstance(p_value, float)
        assert p_value < 0.05  # Should be significant

    def test_two_sample_t_test(self):
        """Test two-sample t-test."""
        np.random.seed(42)
        group1 = np.random.randn(50) + 1.0
        group2 = np.random.randn(50) + 0.0

        t_stat, p_value = two_sample_t_test(group1, group2)

        assert isinstance(t_stat, float)
        assert isinstance(p_value, float)
        assert p_value < 0.05  # Should be significant

    def test_pearson_correlation(self):
        """Test Pearson correlation."""
        np.random.seed(42)
        x = np.random.randn(100)
        y = x * 2 + np.random.randn(100) * 0.1  # Strong correlation

        r, p_value = pearson_correlation(x, y)

        assert isinstance(r, float)
        assert isinstance(p_value, float)
        assert r > 0.9  # Should be highly correlated
        assert p_value < 0.05

    def test_glm_t_test(self):
        """Test GLM t-test."""
        np.random.seed(42)
        n = 50
        X = np.column_stack([np.ones(n), np.random.randn(n)])
        y = X @ np.array([1, 2]) + np.random.randn(n) * 0.5

        t_stat, p_value = glm_t_test(X, y)

        assert isinstance(t_stat, float)
        assert isinstance(p_value, float)


class TestCorrectionMethods:
    """Tests for multiple comparison correction."""

    @pytest.fixture
    def sample_pvalues(self):
        """Create sample p-values."""
        np.random.seed(42)
        return np.random.uniform(0, 0.1, size=100)

    def test_bonferroni_correction(self, sample_pvalues):
        """Test Bonferroni correction."""
        corrected, significant = bonferroni_correction(sample_pvalues, alpha=0.05)

        assert corrected.shape == sample_pvalues.shape
        assert significant.shape == sample_pvalues.shape
        assert np.all(corrected >= sample_pvalues)  # Should be more conservative

    def test_fdr_correction(self, sample_pvalues):
        """Test FDR correction."""
        corrected, significant = fdr_correction(sample_pvalues, alpha=0.05)

        assert corrected.shape == sample_pvalues.shape
        assert significant.shape == sample_pvalues.shape

    def test_fwer_correction(self):
        """Test FWER correction."""
        np.random.seed(42)
        n_perms = 100
        n_tests = 20

        null_distribution = np.random.randn(n_perms, n_tests)
        empirical = np.random.randn(n_tests) + 1.0  # Shifted

        fwer_p, significant = fwer_correction(null_distribution, empirical, alpha=0.05)

        assert fwer_p.shape == (n_tests,)
        assert significant.shape == (n_tests,)


class TestSpecificityTest:
    """Tests for specificity testing."""

    def test_test_against_degree(self):
        """Test degree correlation test."""
        np.random.seed(42)
        n_parcels = 50

        lnm_map = np.random.randn(n_parcels)
        degree_map = lnm_map * 0.8 + np.random.randn(n_parcels) * 0.2  # Correlated

        test = SpecificityTest(n_permutations=100)
        r, p = test.test_against_degree(lnm_map, degree_map)

        assert isinstance(r, float)
        assert isinstance(p, float)
        assert r > 0.5  # Should be correlated

    def test_compute_specificity_index(self):
        """Test specificity index computation."""
        np.random.seed(42)
        n_parcels = 50

        lnm_map = np.random.randn(n_parcels)
        degree_map = lnm_map * 0.5 + np.random.randn(n_parcels) * 0.5

        test = SpecificityTest()
        specificity_idx = test.compute_specificity_index(lnm_map, degree_map)

        assert isinstance(specificity_idx, float)
        assert 0 <= specificity_idx <= 1


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
