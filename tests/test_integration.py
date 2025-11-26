"""
Integration tests for pybrms that require R and brms to be installed.

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
        import pybrms
        
        version = pybrms.get_brms_version()
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
        import pybrms
        
        epilepsy = pybrms.get_brms_data("epilepsy")
        
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
        import pybrms
        
        kidney = pybrms.get_brms_data("kidney")
        assert isinstance(kidney, pd.DataFrame)
        assert len(kidney) > 0
    
    def test_invalid_dataset_raises_error(self):
        """Test that invalid dataset name raises appropriate error"""
        import pybrms
        
        with pytest.raises(Exception):
            # This should fail - dataset doesn't exist
            pybrms.get_brms_data("nonexistent_dataset_name_12345")


@pytest.mark.requires_brms
@pytest.mark.slow
class TestSimpleModelFitting:
    """Test fitting simple models. These are slower tests."""
    
    def test_fit_linear_model_minimal(self, sample_dataframe):
        """Test fitting the simplest possible linear model"""
        import pybrms
        
        # Use minimal iterations for faster testing
        model = pybrms.fit(
            formula="y ~ x1",
            data=sample_dataframe,
            family="gaussian",
            iter=200,
            warmup=100,
            chains=1,
            silent=2,
            refresh=0
        )
        
        # Check return type - now returns brmsfit R object
        import rpy2.robjects as ro
        ro.globalenv['model'] = model
        assert ro.r('class(model)')[0] == 'brmsfit'
        
        # Check we can get parameter names
        import rpy2.robjects as ro
        ro.globalenv['model'] = model
        param_names = list(ro.r('variables(model)'))
        assert len(param_names) > 0
        
        # Check key parameters exist
        assert any('b_Intercept' in p or 'Intercept' in p for p in param_names)
    
    def test_fit_poisson_model(self, poisson_data):
        """Test fitting a Poisson regression model"""
        import pybrms
        
        model = pybrms.fit(
            formula="count ~ predictor",
            data=poisson_data,
            family="poisson",
            iter=200,
            warmup=100,
            chains=1,
            silent=2,
            refresh=0
        )
        
        # Check return type - now returns brmsfit R object
        import rpy2.robjects as ro
        ro.globalenv['model'] = model
        assert ro.r('class(model)')[0] == 'brmsfit'
        
        # Check we can get fixed effects summary
        summary = ro.r('summary(model)$fixed')
        assert summary is not None
    
    def test_fit_with_priors(self, sample_dataframe):
        """Test fitting model with custom priors"""
        import pybrms
        
        model = pybrms.fit(
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
        
        # Check return type - now returns brmsfit R object
        import rpy2.robjects as ro
        ro.globalenv['model'] = model
        assert ro.r('class(model)')[0] == 'brmsfit'
        
        # Check we can get summary
        summary = ro.r('summary(model)$fixed')
        assert summary is not None


@pytest.mark.requires_brms  
@pytest.mark.slow
class TestModelWithRandomEffects:
    """Test models with random effects (more complex)."""
    
    def test_fit_random_intercept(self, sample_dataframe):
        """Test fitting model with random intercepts"""
        import pybrms
        
        # Add more group variation for better convergence
        sample_dataframe['y'] = (
            sample_dataframe['y'] + 
            sample_dataframe['group'].map({'G1': -2, 'G2': 2})
        )
        
        model = pybrms.fit(
            formula="y ~ x1 + (1|group)",
            data=sample_dataframe,
            family="gaussian",
            iter=400,
            warmup=200,
            chains=1,
            silent=2,
            refresh=0
        )
        
        # Check return type - now returns brmsfit R object
        import rpy2.robjects as ro
        ro.globalenv['model'] = model
        assert ro.r('class(model)')[0] == 'brmsfit'
        
        # Check that random effects parameters exist
        param_names = list(ro.r('variables(model)'))
        # Should have standard deviation parameter for random effects
        assert any('sd_group' in p for p in param_names)


@pytest.mark.requires_brms
class TestArVizIntegration:
    """Test integration with arviz for posterior analysis."""
    
    @pytest.mark.slow
    def test_arviz_conversion(self, sample_dataframe):
        """Test that model can be converted to arviz InferenceData"""
        import pybrms
        
        model = pybrms.fit(
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
        
        # Convert to arviz - need to get CSV files from brmsfit
        import rpy2.robjects as ro
        ro.globalenv['model'] = model
        csv_files = list(ro.r('model$output_files()'))
        
        idata = az.from_cmdstan(csv=csv_files)
        
        # Check it's an InferenceData object
        assert isinstance(idata, az.InferenceData)
        
        # Check it has posterior
        assert hasattr(idata, 'posterior')


@pytest.mark.requires_brms
class TestErrorHandling:
    """Test error handling in model fitting."""
    
    def test_invalid_formula_raises_error(self, sample_dataframe):
        """Test that invalid formula raises error"""
        import pybrms
        
        with pytest.raises(Exception):
            # Invalid variable name
            pybrms.fit(
                formula="y ~ nonexistent_variable",
                data=sample_dataframe,
                family="gaussian"
            )
    
    def test_invalid_family_raises_error(self, sample_dataframe):
        """Test that invalid family raises error"""
        import pybrms
        
        with pytest.raises(Exception):
            pybrms.fit(
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
        import pybrms
        
        # Load data
        epilepsy = pybrms.get_brms_data("epilepsy")
        
        # Fit model (with reduced iterations for testing)
        model = pybrms.fit(
            formula="count ~ zAge + zBase * Trt + (1|patient)",
            data=epilepsy,
            family="poisson",
            iter=400,
            warmup=200,
            chains=2,
            silent=2,
            refresh=0
        )
        
        # Check it worked - now returns brmsfit R object
        import rpy2.robjects as ro
        ro.globalenv['model'] = model
        assert ro.r('class(model)')[0] == 'brmsfit'
        
        # Check key parameters exist
        param_names = list(ro.r('variables(model)'))
        assert any('b_zAge' in p for p in param_names)
        assert any('b_zBase' in p for p in param_names)
        
        # Check some basic convergence (Rhat close to 1)
        summary_df = ro.conversion.rpy2py(ro.r('as.data.frame(summary(model)$fixed)'))
        if 'Rhat' in summary_df.columns:
            max_rhat = summary_df['Rhat'].max()
            # Warn if convergence is poor, but don't fail
            # (we're using minimal iterations for speed)
            if max_rhat > 1.1:
                warnings.warn(f"Max R-hat is {max_rhat:.3f} (>1.1) - may need more iterations")


if __name__ == '__main__':
    pytest.main([__file__, '-v', '-m', 'requires_brms'])