"""
Tests for brmspy prediction functions.

Tests all prediction and conversion functions:
- posterior_epred: Expected value predictions
- posterior_predict: Posterior predictive samples
- posterior_linpred: Linear predictor
- log_lik: Log likelihood
- Helper conversion functions
"""

from typing import Any, Callable, cast
import pytest
import pandas as pd
import numpy as np
import warnings


@pytest.mark.requires_brms
class TestPosteriorEpred:
    """Test posterior_epred functionality."""

    @pytest.mark.slow
    def test_epred_basic(self, sample_dataframe):
        """Test basic posterior_epred on fitted model"""
        from brmspy import brms
        import arviz as az

        # Fit a simple model
        model = brms.fit(
            formula="y ~ x1",
            data=sample_dataframe,
            family="gaussian",
            iter=200,
            warmup=100,
            chains=2,
            silent=2,
            refresh=0,
        )

        # Get expected predictions for original data
        epred_result = brms.posterior_epred(model=model, newdata=sample_dataframe)

        # Check result structure
        assert hasattr(epred_result, "idata"), "Should have idata attribute"
        assert hasattr(epred_result, "r"), "Should have r attribute"

        # Check InferenceData structure
        assert isinstance(epred_result.idata, az.InferenceData)
        assert hasattr(epred_result.idata, "posterior")

        # Check data shape
        epred_data = epred_result.idata.posterior["epred"]
        assert (
            len(epred_data.dims) == 3
        ), "Should have 3 dimensions (chain, draw, obs_id)"
        assert epred_data.sizes["chain"] == 2
        assert epred_data.sizes["obs_id"] == len(sample_dataframe)

    @pytest.mark.slow
    def test_epred_newdata(self, sample_dataframe):
        """Test posterior_epred with new data"""
        from brmspy import brms

        # Fit model
        model = brms.fit(
            formula="y ~ x1",
            data=sample_dataframe,
            family="gaussian",
            iter=200,
            warmup=100,
            chains=2,
            silent=2,
            refresh=0,
        )

        # Create new prediction data (fewer obs)
        newdata = sample_dataframe.head(10)

        # Get predictions
        epred_result = brms.posterior_epred(model=model, newdata=newdata)

        # Check prediction shape matches newdata
        epred_data = epred_result.idata.posterior["epred"]
        assert epred_data.sizes["obs_id"] == len(newdata)


@pytest.mark.requires_brms
class TestPosteriorPredict:
    """Test posterior_predict functionality."""

    @pytest.mark.slow
    def test_predict_basic(self, sample_dataframe):
        """Test basic posterior_predict on fitted model"""
        from brmspy import brms
        import arviz as az

        # Fit model
        model = brms.fit(
            formula="y ~ x1",
            data=sample_dataframe,
            family="gaussian",
            iter=200,
            warmup=100,
            chains=2,
            silent=2,
            refresh=0,
        )

        # Get posterior predictions
        pred_result = brms.posterior_predict(model=model, newdata=sample_dataframe)

        # Check result structure
        assert hasattr(pred_result, "idata")
        assert hasattr(pred_result, "r")

        # Check InferenceData structure
        assert isinstance(pred_result.idata, az.InferenceData)
        assert hasattr(pred_result.idata, "posterior_predictive")

        # Check data shape
        pred_data = pred_result.idata.posterior_predictive["y"]
        assert len(pred_data.dims) == 3
        assert pred_data.sizes["chain"] == 2
        assert pred_data.sizes["obs_id"] == len(sample_dataframe)

    @pytest.mark.slow
    def test_predict_without_newdata(self, sample_dataframe):
        """Test posterior_predict without providing newdata"""
        from brmspy import brms

        # Fit model
        model = brms.fit(
            formula="y ~ x1",
            data=sample_dataframe,
            family="gaussian",
            iter=200,
            warmup=100,
            chains=2,
            silent=2,
            refresh=0,
        )

        # Get predictions without newdata (uses original data)
        pred_result = brms.posterior_predict(model=model)

        # Should still work
        assert hasattr(pred_result, "idata")
        assert hasattr(pred_result.idata, "posterior_predictive")


@pytest.mark.requires_brms
class TestPosteriorLinpred:
    """Test posterior_linpred functionality."""

    @pytest.mark.slow
    def test_linpred_basic(self, sample_dataframe):
        """Test basic posterior_linpred on fitted model"""
        from brmspy import brms
        import arviz as az

        # Fit model
        model = brms.fit(
            formula="y ~ x1",
            data=sample_dataframe,
            family="gaussian",
            iter=200,
            warmup=100,
            chains=2,
            silent=2,
            refresh=0,
        )

        # Get linear predictions
        linpred_result = brms.posterior_linpred(model=model, newdata=sample_dataframe)

        # Check result structure
        assert hasattr(linpred_result, "idata")
        assert hasattr(linpred_result, "r")

        # Check InferenceData structure
        assert isinstance(linpred_result.idata, az.InferenceData)
        assert hasattr(linpred_result.idata, "predictions")

        # Check data shape
        linpred_data = linpred_result.idata.predictions["linpred"]
        assert len(linpred_data.dims) == 3
        assert linpred_data.sizes["chain"] == 2


@pytest.mark.requires_brms
class TestLogLik:
    """Test log_lik functionality."""

    @pytest.mark.slow
    def test_log_lik_basic(self, sample_dataframe):
        """Test basic log_lik on fitted model"""
        from brmspy import brms
        import arviz as az

        # Fit model
        model = brms.fit(
            formula="y ~ x1",
            data=sample_dataframe,
            family="gaussian",
            iter=200,
            warmup=100,
            chains=2,
            silent=2,
            refresh=0,
        )

        # Get log likelihood
        ll_result = brms.log_lik(model=model, newdata=sample_dataframe)

        # Check result structure
        assert hasattr(ll_result, "idata")
        assert hasattr(ll_result, "r")

        # Check InferenceData structure
        assert isinstance(ll_result.idata, az.InferenceData)
        assert hasattr(ll_result.idata, "log_likelihood")


@pytest.mark.requires_brms
class TestBrmsfitToIdata:
    """Test brmsfit_to_idata comprehensive conversion."""

    @pytest.mark.slow
    def test_complete_idata_conversion(self, sample_dataframe):
        """Test that brmsfit_to_idata creates all groups"""
        from brmspy import brms

        # Fit model
        model = brms.fit(
            formula="y ~ x1",
            data=sample_dataframe,
            family="gaussian",
            iter=200,
            warmup=100,
            chains=2,
            silent=2,
            refresh=0,
        )

        idata = model.idata

        # Check all expected groups are present
        assert hasattr(idata, "posterior"), "Should have posterior group"
        assert hasattr(
            idata, "posterior_predictive"
        ), "Should have posterior_predictive"
        assert hasattr(idata, "log_likelihood"), "Should have log_likelihood"
        assert hasattr(idata, "observed_data"), "Should have observed_data"

        # Check observed data
        assert "y" in idata.observed_data
        assert len(idata.observed_data["y"]) == len(sample_dataframe)

    @pytest.mark.slow
    def test_posterior_predictive_shape(self, sample_dataframe):
        """Test posterior predictive has correct shape"""
        from brmspy import brms

        # Fit model with known parameters
        n_chains = 2
        n_draws = 100

        model = brms.fit(
            formula="y ~ x1",
            data=sample_dataframe,
            family="gaussian",
            iter=200,
            warmup=100,
            chains=n_chains,
            silent=2,
            refresh=0,
        )
        idata = model.idata

        # Check posterior predictive shape
        pp = idata.posterior_predictive["y"]
        assert pp.sizes["chain"] == n_chains
        assert pp.sizes["draw"] == n_draws
        assert pp.sizes["obs_id"] == len(sample_dataframe)


@pytest.mark.requires_brms
class TestConversionHelpers:
    """Test helper conversion functions."""

    @pytest.mark.slow
    def test_reshape_r_prediction_to_arviz(self, sample_dataframe):
        """Test _reshape_r_prediction_to_arviz function"""
        from brmspy import brms
        import arviz as az
        from xarray.core.coordinates import DatasetCoordinates

        # Fit model
        model = brms.fit(
            formula="y ~ x1",
            data=sample_dataframe,
            family="gaussian",
            iter=200,
            warmup=100,
            chains=2,
            silent=2,
            refresh=0,
        )
        posterior_predict = brms.posterior_predict(model)
        idata = posterior_predict.idata

        var_name = list(idata.posterior_predictive.data_vars)[0]

        # Extract as an xarray.DataArray
        da = idata.posterior_predictive[var_name]

        # Convert to a plain NumPy array
        reshaped_data = da.values

        coords, dims = (
            idata.posterior_predictive.coords,
            idata.posterior_predictive.dims,
        )

        # Check output structure
        assert isinstance(reshaped_data, np.ndarray)
        assert reshaped_data.ndim == 3, "Should be 3D array (chains, draws, obs)"

        assert isinstance(coords, DatasetCoordinates)
        assert "chain" in coords
        assert "draw" in coords
        assert "obs_id" in coords

        assert list(dims) == ["chain", "draw", "obs_id"]

    @pytest.mark.slow
    def test_epred_to_idata_helper(self, sample_dataframe):
        """Test brms_epred_to_idata helper function"""
        from brmspy import brms

        # Fit model
        model = brms.fit(
            formula="y ~ x1",
            data=sample_dataframe,
            family="gaussian",
            iter=200,
            warmup=100,
            chains=2,
            silent=2,
            refresh=0,
        )

        idata = brms.posterior_epred(model).idata

        # Check structure
        assert hasattr(idata, "posterior")
        assert "epred" in cast(Any, idata).posterior


@pytest.mark.requires_brms
class TestPredictionConsistency:
    """Test consistency between prediction methods."""

    @pytest.mark.slow
    def test_epred_vs_predict_difference(self, sample_dataframe):
        """
        Test that posterior_epred and posterior_predict give different results.

        Epred should give expected value (mean), while predict adds noise.
        Therefore predict should have higher variance.
        """
        from brmspy import brms

        # Fit model
        model = brms.fit(
            formula="y ~ x1",
            data=sample_dataframe,
            family="gaussian",
            iter=400,
            warmup=200,
            chains=2,
            silent=2,
            refresh=0,
        )

        # Get both types of predictions
        epred = brms.posterior_epred(model=model, newdata=sample_dataframe)
        predict = brms.posterior_predict(model=model, newdata=sample_dataframe)

        # Extract values
        epred_vals = epred.idata.posterior["epred"].values
        predict_vals = predict.idata.posterior_predictive["y"].values

        # Predictions should have higher variance (they include observation noise)
        epred_std = np.std(epred_vals)
        predict_std = np.std(predict_vals)

        assert predict_std > epred_std, (
            f"Predictions (std={predict_std:.3f}) should have higher variance "
            f"than expected values (std={epred_std:.3f})"
        )


@pytest.mark.requires_brms
class TestPoissonPredictions:
    """Test predictions work with non-Gaussian families."""

    @pytest.mark.slow
    def test_poisson_predictions(self, poisson_data):
        """Test prediction functions with Poisson family"""
        from brmspy import brms

        # Fit Poisson model
        model = brms.fit(
            formula="count ~ predictor",
            data=poisson_data,
            family="poisson",
            iter=200,
            warmup=100,
            chains=2,
            silent=2,
            refresh=0,
        )

        # All prediction types should work
        epred = brms.posterior_epred(model=model, newdata=poisson_data)
        predict = brms.posterior_predict(model=model, newdata=poisson_data)
        linpred = brms.posterior_linpred(model=model, newdata=poisson_data)

        # Check they all have correct structure
        assert hasattr(epred.idata, "posterior")
        assert hasattr(predict.idata, "posterior_predictive")
        assert hasattr(linpred.idata, "predictions")

        # For Poisson: predictions should be integers
        predict_vals = predict.idata.posterior_predictive["y"].values
        # Check at least some values are non-zero integers
        assert np.any(predict_vals > 0)
        # Poisson predictions should be close to integers
        # (allowing for floating point representation)
        assert np.allclose(predict_vals, np.round(predict_vals), atol=1e-10)


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-m", "requires_brms"])
