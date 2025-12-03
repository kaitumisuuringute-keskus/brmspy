"""
Tests for prior functions (get_prior, default_prior).

These tests verify that prior specification retrieval works correctly
for various model formulas and families.
"""
import pytest
import pandas as pd


@pytest.mark.requires_brms
class TestGetPrior:
    """Test get_prior() and default_prior() functions."""
    
    def test_get_prior_basic_structure(self, sample_dataframe):
        """Test get_prior returns DataFrame with expected structure for simple formula"""
        from brmspy import brms
        
        # Get priors for simple model
        priors = brms.get_prior(
            formula="y ~ x1",
            data=sample_dataframe,
            family="gaussian"
        )
        
        # Verify return type
        assert isinstance(priors, pd.DataFrame)
        
        # Verify essential columns exist
        expected_cols = {'prior', 'class', 'coef', 'group', 'source'}
        assert expected_cols.issubset(set(priors.columns))
        
        # Verify has rows for key parameter classes
        assert 'b' in priors['class'].values  # Fixed effects
        assert 'Intercept' in priors['class'].values  # Intercept
        assert 'sigma' in priors['class'].values  # Residual SD for gaussian
        
        # Verify source column indicates defaults
        assert 'default' in priors['source'].values
        
        # Verify non-empty DataFrame
        assert len(priors) > 0
    
    def test_get_prior_complex_formula(self, sample_dataframe):
        """Test get_prior with complex formula including random effects and interactions"""
        from brmspy import brms
        
        # Complex formula with random effects and interaction
        priors = brms.get_prior(
            formula="y ~ x1 * x2 + (1|group)",
            data=sample_dataframe,
            family="gaussian"
        )
        
        # Verify return type
        assert isinstance(priors, pd.DataFrame)
        
        # Verify has rows for fixed effects
        b_rows = priors[priors['class'] == 'b']
        assert len(b_rows) > 0
        
        # Verify interaction term appears
        assert any('x1:x2' in str(coef) for coef in priors['coef'].values)
        
        # Verify random effects appear
        assert 'sd' in priors['class'].values  # SD of random effects
        assert 'group' in priors['group'].values  # Group variable
        
        # Verify structure matches expected output format
        # Should have rows for: Intercept, b class, individual coeffs, sd for group
        sd_rows = priors[priors['class'] == 'sd']
        assert len(sd_rows) > 0
        assert any(priors['group'] == 'group')
    
    def test_default_prior_equivalence(self, sample_dataframe):
        """Test default_prior returns same results as get_prior for same inputs"""
        from brmspy import brms
        
        formula = "y ~ x1 + x2"
        
        # Get priors using get_prior
        priors_get = brms.get_prior(
            formula=formula,
            data=sample_dataframe,
            family="gaussian"
        )
        
        # Get priors using default_prior
        priors_default = brms.default_prior(
            formula,
            data=sample_dataframe,
            family="gaussian"
        )
        
        # Both should return DataFrames
        assert isinstance(priors_get, pd.DataFrame)
        assert isinstance(priors_default, pd.DataFrame)
        
        # Should have same shape
        assert priors_get.shape == priors_default.shape
        
        # Should have same parameter classes
        assert set(priors_get['class'].values) == set(priors_default['class'].values)
        
        # Should have same coefficients
        get_coefs = set(str(c) for c in priors_get['coef'].values)
        default_coefs = set(str(c) for c in priors_default['coef'].values)
        assert get_coefs == default_coefs
    
    def test_get_prior_different_families(self, sample_dataframe):
        """Test get_prior with different distribution families returns appropriate priors"""
        from brmspy import brms
        
        # Test Gaussian family
        gaussian_priors = brms.get_prior(
            formula="y ~ x1",
            data=sample_dataframe,
            family="gaussian"
        )
        
        # Gaussian should have sigma parameter
        assert 'sigma' in gaussian_priors['class'].values
        
        # Test Poisson family (no sigma, uses log link)
        # Create count data
        import numpy as np
        count_data = sample_dataframe.copy()
        count_data['count'] = np.random.poisson(5, size=len(count_data))
        
        poisson_priors = brms.get_prior(
            formula="count ~ x1",
            data=count_data,
            family="poisson"
        )
        
        # Poisson should NOT have sigma parameter
        assert 'sigma' not in poisson_priors['class'].values
        
        # Both should have Intercept and b class
        for priors in [gaussian_priors, poisson_priors]:
            assert 'Intercept' in priors['class'].values
            assert 'b' in priors['class'].values
        
        # Verify different families have different parameter structures
        gaussian_classes = set(gaussian_priors['class'].values)
        poisson_classes = set(poisson_priors['class'].values)
        assert gaussian_classes != poisson_classes


if __name__ == '__main__':
    pytest.main([__file__, '-v'])