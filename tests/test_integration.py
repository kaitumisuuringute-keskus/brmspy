"""
Integration tests for brmspy that require R and brms to be installed.

These tests check end-to-end functionality:
- brms installation
- Data loading from brms
- Model fitting with simple examples
- CmdStanPy integration

Mark with: @pytest.mark.requires_brms
These will be automatically skipped if brms is not installed.
"""
import pytest
import pandas as pd
import numpy as np
import warnings


@pytest.mark.requires_brms
class TestBrmsInstallation:
    """Test brms installation and version checking."""
    
    def test_get_brms_version(self):
        """Test that we can get brms version"""
        import brmspy
        
        version = brmspy.get_brms_version()
        assert isinstance(version, str)
        assert len(version) > 0
        # Version should be like "2.21.0" or similar
        assert version.count('.') >= 1
    
    def test_brms_import_successful(self):
        """Test that brms can be imported via rpy2"""
        import rpy2.robjects.packages as rpackages
        brms = rpackages.importr("brms")
        assert brms is not None


@pytest.mark.requires_brms
class TestDataLoading:
    """Test loading example datasets from brms."""
    
    def test_get_epilepsy_data(self):
        """Test loading the epilepsy dataset"""
        import brmspy
        
        epilepsy = brmspy.get_brms_data("epilepsy")
        
        # Check it's a DataFrame
        assert isinstance(epilepsy, pd.DataFrame)
        
        # Check it has expected columns
        assert 'count' in epilepsy.columns
        assert 'patient' in epilepsy.columns
        
        # Check data shape
        assert len(epilepsy) > 0
        assert len(epilepsy.columns) > 0
    
    def test_get_kidney_data(self):
        """Test loading the kidney dataset"""
        import brmspy
        
        kidney = brmspy.get_brms_data("kidney")
        assert isinstance(kidney, pd.DataFrame)
        assert len(kidney) > 0
    
    def test_invalid_dataset_raises_error(self):
        """Test that invalid dataset name raises appropriate error"""
        import brmspy
        
        with pytest.raises(Exception):
            # This should fail - dataset doesn't exist
            brmspy.get_brms_data("nonexistent_dataset_name_12345")


@pytest.mark.requires_brms
@pytest.mark.slow
class TestSimpleModelFitting:
    """Test fitting simple models. These are slower tests."""
    
    def test_fit_linear_model_minimal(self, sample_dataframe):
        """Test fitting the simplest possible linear model"""
        import brmspy
        
        # Use minimal iterations for faster testing
        model = brmspy.fit(
            formula="y ~ x1",
            data=sample_dataframe,
            family="gaussian",
            iter=200,
            warmup=100,
            chains=1,
            silent=2,
            refresh=0
        )
        
        # Check return type - now returns arviz InferenceData by default
        import arviz as az
        assert isinstance(model, az.InferenceData)
        
        # Check we can get parameter names
        param_names = list(model.posterior.data_vars)
        assert len(param_names) > 0
        
        # Check key parameters exist
        assert any('b_Intercept' in p or 'Intercept' in p for p in param_names)
    
    def test_fit_poisson_model(self, poisson_data):
        """Test fitting a Poisson regression model"""
        import brmspy
        
        model = brmspy.fit(
            formula="count ~ predictor",
            data=poisson_data,
            family="poisson",
            iter=200,
            warmup=100,
            chains=1,
            silent=2,
            refresh=0
        )
        
        # Check return type - now returns arviz InferenceData by default
        import arviz as az
        assert isinstance(model, az.InferenceData)
        
        # Check we can get summary
        summary = az.summary(model)
        assert summary is not None
        assert len(summary) > 0
    
    def test_fit_with_priors(self, sample_dataframe):
        """Test fitting model with custom priors"""
        import brmspy
        
        model = brmspy.fit(
            formula="y ~ x1",
            data=sample_dataframe,
            priors=[("normal(0, 5)", "b")],
            family="gaussian",
            iter=200,
            warmup=100,
            chains=1,
            silent=2,
            refresh=0
        )
        
        # Check return type - now returns arviz InferenceData by default
        import arviz as az
        assert isinstance(model, az.InferenceData)
        
        # Check we can get summary
        summary = az.summary(model)
        assert summary is not None


@pytest.mark.requires_brms  
@pytest.mark.slow
class TestModelWithRandomEffects:
    """Test models with random effects (more complex)."""
    
    def test_fit_random_intercept(self, sample_dataframe):
        """Test fitting model with random intercepts"""
        import brmspy
        
        # Add more group variation for better convergence
        sample_dataframe['y'] = (
            sample_dataframe['y'] + 
            sample_dataframe['group'].map({'G1': -2, 'G2': 2})
        )
        
        model = brmspy.fit(
            formula="y ~ x1 + (1|group)",
            data=sample_dataframe,
            family="gaussian",
            iter=400,
            warmup=200,
            chains=1,
            silent=2,
            refresh=0
        )
        
        # Check return type - now returns arviz InferenceData by default
        import arviz as az
        assert isinstance(model, az.InferenceData)
        
        # Check that random effects parameters exist
        param_names = list(model.posterior.data_vars)
        # Should have standard deviation parameter for random effects
        assert any('sd_group' in p for p in param_names)


@pytest.mark.requires_brms
class TestArVizIntegration:
    """Test integration with arviz for posterior analysis."""
    
    @pytest.mark.slow
    def test_arviz_conversion(self, sample_dataframe):
        """Test that model can be converted to arviz InferenceData"""
        import brmspy
        
        model = brmspy.fit(
            formula="y ~ x1",
            data=sample_dataframe,
            iter=200,
            warmup=100,
            chains=1,
            silent=2,
            refresh=0
        )
        
        try:
            import arviz as az
        except ImportError:
            pytest.skip("arviz not installed")
        
        # Model is already InferenceData, no conversion needed
        assert isinstance(model, az.InferenceData)
        
        # Check it has posterior
        assert hasattr(model, 'posterior')


@pytest.mark.requires_brms
class TestErrorHandling:
    """Test error handling in model fitting."""
    
    def test_invalid_formula_raises_error(self, sample_dataframe):
        """Test that invalid formula raises error"""
        import brmspy
        
        with pytest.raises(Exception):
            # Invalid variable name
            brmspy.fit(
                formula="y ~ nonexistent_variable",
                data=sample_dataframe,
                family="gaussian"
            )
    
    def test_invalid_family_raises_error(self, sample_dataframe):
        """Test that invalid family raises error"""
        import brmspy
        
        with pytest.raises(Exception):
            brmspy.fit(
                formula="y ~ x1",
                data=sample_dataframe,
                family="not_a_real_family"
            )


@pytest.mark.requires_brms
class TestRealWorldExample:
    """Test with real brms example dataset (epilepsy)."""
    
    @pytest.mark.slow
    def test_epilepsy_example(self):
        """Test the epilepsy example from README"""
        import brmspy
        
        # Load data
        epilepsy = brmspy.get_brms_data("epilepsy")
        
        # Fit model (with reduced iterations for testing)
        model = brmspy.fit(
            formula="count ~ zAge + zBase * Trt + (1|patient)",
            data=epilepsy,
            family="poisson",
            iter=400,
            warmup=200,
            chains=2,
            silent=2,
            refresh=0
        )
        
        # Check it worked - now returns arviz InferenceData by default
        import arviz as az
        assert isinstance(model, az.InferenceData)
        
        # Check key parameters exist
        param_names = list(model.posterior.data_vars)
        assert any('b_zAge' in p for p in param_names)
        assert any('b_zBase' in p for p in param_names)
        
        # Check some basic convergence (Rhat close to 1)
        summary = az.summary(model)
        if 'r_hat' in summary.columns:
            max_rhat = summary['r_hat'].max()
            # Warn if convergence is poor, but don't fail
            # (we're using minimal iterations for speed)
            if max_rhat > 1.1:
                warnings.warn(f"Max R-hat is {max_rhat:.3f} (>1.1) - may need more iterations")


@pytest.mark.requires_brms
class TestNaNRegression:
    """Regression tests for specific bugs that were fixed."""
    
    @pytest.mark.slow
    def test_no_nans_in_idata_conversion(self):
        """
        Regression test for NaN bug in _brmsfit_to_idata().
        
        The posterior R package numbers draws sequentially across chains
        (chain1: 1-500, chain2: 501-1000), but arviz expects draws numbered
        within each chain (each chain: 0-499). This test verifies that the
        conversion correctly renumbers draws to avoid NaNs.
        
        Bug was: df.pivot(index='.draw', columns='.chain', values=col)
        Fix: Renumber draws within each chain before pivoting
        """
        import brmspy
        
        # Create simple test data
        np.random.seed(42)
        data = pd.DataFrame({
            'y': np.random.randn(50),
            'x': np.random.randn(50)
        })
        
        # Fit model with return_type="both" to check both formats
        result = brmspy.fit(
            formula="y ~ x",
            data=data,
            family="gaussian",
            return_type="both",
            chains=4,
            iter=200,
            warmup=100,
            silent=2,
            refresh=0
        )
        
        # Check that brmsfit has no NaNs (via posterior package)
        from rpy2.robjects.packages import importr
        from rpy2.robjects import pandas2ri, default_converter
        from rpy2.robjects.conversion import localconverter
        
        posterior = importr('posterior')
        draws = posterior.as_draws_df(result.brmsfit)
        
        with localconverter(default_converter + pandas2ri.converter):
            df = pandas2ri.rpy2py(draws)
        
        # Verify no NaNs in original draws from R
        assert not df.isna().any().any(), "brmsfit draws should not contain NaNs"
        
        # Check that InferenceData has no NaNs (this was the bug)
        idata = result.idata
        
        # Check all parameters in posterior group
        for param_name in idata.posterior.data_vars:
            param_values = idata.posterior[param_name].values
            n_nans = np.isnan(param_values).sum()
            assert n_nans == 0, (
                f"Parameter '{param_name}' has {n_nans} NaN values in InferenceData. "
                f"This indicates the draw renumbering fix failed."
            )
        
        # Verify we have the expected shape (chains, draws)
        # Use .sizes instead of .dims to avoid FutureWarning
        assert idata.posterior.sizes['chain'] == 4, "Should have 4 chains"
        assert idata.posterior.sizes['draw'] == 100, "Should have 100 draws per chain"
        
        print("✓ No NaNs found in InferenceData conversion")


if __name__ == '__main__':
    pytest.main([__file__, '-v', '-m', 'requires_brms'])