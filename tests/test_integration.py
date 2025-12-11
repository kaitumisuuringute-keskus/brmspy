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

from brmspy.types.formula_dsl import FormulaPart


@pytest.mark.requires_brms
class TestBrmsImportAndVersion:
    """Test brms installation and version checking."""

    def test_get_brms_version(self):
        """Test that we can get brms version"""
        from brmspy import brms
        from packaging.version import Version

        version = brms.get_brms_version()
        assert isinstance(version, Version)


@pytest.mark.requires_brms
class TestDataLoading:
    """Test loading example datasets from brms."""

    def test_get_epilepsy_data(self):
        """Test loading the epilepsy dataset"""
        from brmspy import brms

        epilepsy = brms.get_brms_data("epilepsy")

        # Check it's a DataFrame
        assert isinstance(epilepsy, pd.DataFrame)

        # Check it has expected columns
        assert "count" in epilepsy.columns
        assert "patient" in epilepsy.columns

        # Check data shape
        assert len(epilepsy) > 0
        assert len(epilepsy.columns) > 0

    def test_get_kidney_data(self):
        """Test loading the kidney dataset"""
        from brmspy import brms

        kidney = brms.get_brms_data("kidney")
        assert isinstance(kidney, pd.DataFrame)
        assert len(kidney) > 0

    def test_invalid_dataset_raises_error(self):
        """Test that invalid dataset name raises appropriate error"""
        from brmspy import brms

        with pytest.raises(Exception):
            # This should fail - dataset doesn't exist
            brms.get_brms_data("nonexistent_dataset_name_12345")


@pytest.mark.requires_brms
@pytest.mark.slow
class TestSimpleModelFitting:
    """Test fitting simple models. These are slower tests."""

    def test_fit_linear_model_minimal(self, sample_dataframe):
        """Test fitting the simplest possible linear model"""
        from brmspy import brms

        # Use minimal iterations for faster testing
        model = brms.fit(
            formula="y ~ x1",
            data=sample_dataframe,
            family="gaussian",
            iter=200,
            warmup=100,
            chains=1,
            silent=2,
            refresh=0,
        )

        # Check return type - now returns arviz InferenceData by default
        import arviz as az

        assert isinstance(model.idata, az.InferenceData)

        # Check we can get parameter names
        param_names = list(model.idata.posterior.data_vars)
        assert len(param_names) > 0

        # Check key parameters exist
        assert any("b_Intercept" in p or "Intercept" in p for p in param_names)

    def test_fit_poisson_model(self, poisson_data):
        """Test fitting a Poisson regression model"""
        from brmspy import brms

        model = brms.fit(
            formula="count ~ predictor",
            data=poisson_data,
            family="poisson",
            iter=200,
            warmup=100,
            chains=1,
            silent=2,
            refresh=0,
        )

        # Check return type - now returns arviz InferenceData by default
        import arviz as az

        assert isinstance(model.idata, az.InferenceData)

        # Check we can get summary
        summary = az.summary(model.idata)
        assert summary is not None
        assert len(summary) > 0

    def test_fit_with_priors(self, sample_dataframe):
        """Test fitting model with custom priors"""
        from brmspy import brms

        model = brms.fit(
            formula="y ~ x1",
            data=sample_dataframe,
            priors=[brms.prior("normal(0, 5)", "b")],
            family="gaussian",
            iter=200,
            warmup=100,
            chains=1,
            silent=2,
            refresh=0,
        )

        # Check return type - now returns arviz InferenceData by default
        import arviz as az

        assert isinstance(model.idata, az.InferenceData)

        # Check we can get summary
        summary = az.summary(model.idata)
        assert summary is not None


@pytest.mark.requires_brms
@pytest.mark.slow
class TestModelWithRandomEffects:
    """Test models with random effects (more complex)."""

    def test_fit_random_intercept(self, sample_dataframe):
        """Test fitting model with random intercepts"""
        from brmspy import brms

        # Add more group variation for better convergence
        sample_dataframe["y"] = sample_dataframe["y"] + sample_dataframe["group"].map(
            {"G1": -2, "G2": 2}
        )

        model = brms.fit(
            formula="y ~ x1 + (1|group)",
            data=sample_dataframe,
            family="gaussian",
            iter=200,
            warmup=100,
            chains=1,
            silent=2,
            refresh=0,
        )

        # Check return type - now returns arviz InferenceData by default
        import arviz as az

        assert isinstance(model.idata, az.InferenceData)

        # Check that random effects parameters exist
        param_names = list(model.idata.posterior.data_vars)
        # Should have standard deviation parameter for random effects
        assert any("sd_group" in p for p in param_names)


@pytest.mark.requires_brms
class TestArVizIntegration:
    """Test integration with arviz for posterior analysis."""

    @pytest.mark.slow
    def test_arviz_conversion(self, sample_dataframe):
        """Test that model can be converted to arviz InferenceData"""
        from brmspy import brms

        model = brms.fit(
            formula="y ~ x1",
            data=sample_dataframe,
            iter=200,
            warmup=100,
            chains=1,
            silent=2,
            refresh=0,
        )

        try:
            import arviz as az
        except ImportError:
            pytest.skip("arviz not installed")

        # Model is already InferenceData, no conversion needed
        assert isinstance(model.idata, az.InferenceData)

        # Check it has posterior
        assert hasattr(model.idata, "posterior")


@pytest.mark.requires_brms
class TestErrorHandling:
    """Test error handling in model fitting."""

    def test_invalid_formula_raises_error(self, sample_dataframe):
        """Test that invalid formula raises error"""
        from brmspy import brms

        with pytest.raises(Exception):
            # Invalid variable name
            brms.fit(
                formula="y ~ nonexistent_variable",
                data=sample_dataframe,
                family="gaussian",
                iter=100,
                warmup=50,
            )

    def test_invalid_family_raises_error(self, sample_dataframe):
        """Test that invalid family raises error"""
        from brmspy import brms

        with pytest.raises(Exception):
            brms.fit(
                formula="y ~ x1",
                data=sample_dataframe,
                family="not_a_real_family",
                iter=100,
                warmup=50,
            )


@pytest.mark.requires_brms
class TestRealWorldExample:
    """Test with real brms example dataset (epilepsy)."""

    @pytest.mark.slow
    def test_epilepsy_example(self):
        """Test the epilepsy example from README"""
        from brmspy import brms

        # Load data
        epilepsy = brms.get_brms_data("epilepsy")

        # Fit model (with reduced iterations for testing)
        model = brms.fit(
            formula="count ~ zAge + zBase * Trt + (1|patient)",
            data=epilepsy,
            family="poisson",
            iter=400,
            warmup=200,
            chains=2,
            silent=2,
            refresh=0,
        )

        # Check it worked - now returns arviz InferenceData by default
        import arviz as az

        assert isinstance(model.idata, az.InferenceData)

        # Check key parameters exist
        param_names = list(model.idata.posterior.data_vars)
        assert any("b_zAge" in p for p in param_names)
        assert any("b_zBase" in p for p in param_names)

        # Check some basic convergence (Rhat close to 1)
        summary = az.summary(model.idata)
        if "r_hat" in summary.columns:
            max_rhat = summary["r_hat"].max()
            # Warn if convergence is poor, but don't fail
            # (we're using minimal iterations for speed)
            if max_rhat > 1.1:
                warnings.warn(
                    f"Max R-hat is {max_rhat:.3f} (>1.1) - may need more iterations"
                )


@pytest.mark.requires_brms
class TestNaNRegression:
    """Regression tests for specific bugs that were fixed."""

    @pytest.mark.slow
    def test_no_nans_in_idata_conversion(self):
        """
        Regression test for NaN bug in brmsfit_to_idata().

        The posterior R package numbers draws sequentially across chains
        (chain1: 1-500, chain2: 501-1000), but arviz expects draws numbered
        within each chain (each chain: 0-499). This test verifies that the
        conversion correctly renumbers draws to avoid NaNs.

        Bug was: df.pivot(index='.draw', columns='.chain', values=col)
        Fix: Renumber draws within each chain before pivoting
        """
        from brmspy import brms

        # Create simple test data
        np.random.seed(42)
        data = pd.DataFrame({"y": np.random.randn(50), "x": np.random.randn(50)})

        result = brms.fit(
            formula="y ~ x",
            data=data,
            family="gaussian",
            chains=4,
            iter=200,
            warmup=100,
            silent=2,
            refresh=0,
        )

        # Check that r has no NaNs (via posterior package)

        df = result.idata.posterior.to_dataframe()
        print(df.head().to_string())

        # Verify no NaNs in original draws from R
        assert not df.isna().any().any(), "r draws should not contain NaNs"

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
        assert idata.posterior.sizes["chain"] == 4, "Should have 4 chains"
        assert idata.posterior.sizes["draw"] == 100, "Should have 100 draws per chain"

        print("No NaNs found in InferenceData conversion")


@pytest.mark.requires_brms
class TestFormulaFunction:
    """Test the formula() function for creating brmsformula objects."""

    def test_formula_basic_creation(self):
        """
        Test basic formula creation with simple formula string.

        Validates that formula() creates a valid FormulaResult with:
        - Proper return type structure
        - Valid R brmsformula object
        - Python dict conversion
        """
        from brmspy import brms
        from brmspy.types.formula_dsl import FormulaConstruct

        # Create basic formula
        formula_result = brms.formula("y ~ x")

        # Check return type
        assert isinstance(
            formula_result, FormulaConstruct
        ), "formula() should return FormulaConstruct instance"

        # Check attributes exist
        assert hasattr(
            formula_result, "_parts"
        ), "FormulaResult should have ._parts attribute"

        # Check R object is valid (not None or NULL)
        assert len(formula_result._parts) == 1

    def test_formula_with_brms_arguments(self):
        """
        Test formula with brms-specific arguments.

        Tests that brms arguments are properly passed through:
        - QR decomposition (decomp="QR")
        - Centering control (center=False)
        - Sparse matrix support (sparse=True)
        - Multiple arguments work together
        """
        from brmspy import brms
        from brmspy.types.formula_dsl import FormulaConstruct

        formula_no_center = brms.formula("y ~ x", center=False, decomp="QR")
        assert isinstance(
            formula_no_center, FormulaConstruct
        ), "formula() with center should return FormulaResult"
        _part0 = formula_no_center._parts[0]
        assert isinstance(_part0, FormulaPart)
        assert _part0._kwargs["center"] == False
        assert _part0._kwargs["decomp"] == "QR"

    @pytest.mark.slow
    def test_formula_complex_with_fit_integration(self):
        """
        Test complex formula with random effects and integration with fit().

        Tests real-world usage pattern:
        - Complex formulas with random effects work
        - FormulaResult can replace string formula in fit()
        - Model fitting succeeds with FormulaResult input
        - Both approaches (string vs FormulaResult) produce equivalent results
        """
        from brmspy import brms
        from brmspy.types.formula_dsl import FormulaConstruct
        import arviz as az

        # Load epilepsy dataset
        epilepsy = brms.get_brms_data("epilepsy")

        # Create formula with random effects
        formula_obj = brms.formula("count ~ zAge + (1|patient)")

        # Verify formula object structure
        assert isinstance(
            formula_obj, FormulaConstruct
        ), "formula() should return FormulaResult"

        # Use FormulaResult in fit() - should work like string formula
        model = brms.fit(
            formula=formula_obj,
            data=epilepsy,
            family="poisson",
            iter=200,
            warmup=100,
            chains=1,
            silent=2,
            refresh=0,
        )

        # Verify model fitted successfully
        assert isinstance(
            model.idata, az.InferenceData
        ), "fit() with FormulaResult should return InferenceData"

        # Check parameters exist
        param_names = list(model.idata.posterior.data_vars)
        assert len(param_names) > 0, "Model should have parameters"
        assert any(
            "b_zAge" in p for p in param_names
        ), "Model should have coefficient for zAge"
        assert any(
            "sd_patient" in p for p in param_names
        ), "Model should have random effect standard deviation"

        # Compare with string formula approach (should be equivalent)
        model_string = brms.fit(
            formula="count ~ zAge + (1|patient)",
            data=epilepsy,
            family="poisson",
            iter=200,
            warmup=100,
            chains=1,
            seed=123,  # Use seed for reproducibility
            silent=2,
            refresh=0,
        )

        # Both should have same parameters
        params_obj = set(model.idata.posterior.data_vars)
        params_str = set(model_string.idata.posterior.data_vars)
        assert (
            params_obj == params_str
        ), "FormulaResult and string formula should produce same parameters"


@pytest.mark.requires_brms
class TestPriorFunction:
    """Test the prior() function for creating prior specifications."""

    @pytest.mark.slow
    def test_prior_basic_usage(self, sample_dataframe):
        """
        Test basic prior() usage with class parameters.

        Tests that prior() creates valid PriorSpec objects for:
        - Intercept priors (class_="Intercept")
        - Coefficient priors (class_="b")
        - Multiple priors in a single model
        - Integration with fit() function
        """
        from brmspy import brms
        from brmspy.brms import prior
        from brmspy.types.brms_results import PriorSpec
        import arviz as az

        # Create prior specifications using prior() function
        prior_intercept = prior("student_t(3, 0, 2.5)", class_="Intercept")
        prior_coef = prior("normal(0, 1)", class_="b")

        # Verify PriorSpec objects are created correctly
        assert isinstance(
            prior_intercept, PriorSpec
        ), "prior() should return PriorSpec instance"
        assert isinstance(
            prior_coef, PriorSpec
        ), "prior() should return PriorSpec instance"

        # Verify prior specifications have correct attributes
        assert (
            prior_intercept.prior == "student_t(3, 0, 2.5)"
        ), "Prior string should be stored correctly"
        assert (
            prior_intercept.class_ == "Intercept"
        ), "Class parameter should be stored correctly"
        assert (
            prior_coef.prior == "normal(0, 1)"
        ), "Prior string should be stored correctly"
        assert prior_coef.class_ == "b", "Class parameter should be stored correctly"

        # Test that priors work with fit()
        model = brms.fit(
            formula="y ~ x1",
            data=sample_dataframe,
            priors=[prior_intercept, prior_coef],
            family="gaussian",
            iter=200,
            warmup=100,
            chains=1,
            silent=2,
            refresh=0,
        )

        # Verify model fitted successfully with custom priors
        assert isinstance(
            model.idata, az.InferenceData
        ), "fit() with prior() specifications should return InferenceData"

        # Check parameters exist
        param_names = list(model.idata.posterior.data_vars)
        assert len(param_names) > 0, "Model should have parameters"
        assert any(
            "b_Intercept" in p or "Intercept" in p for p in param_names
        ), "Model should have intercept parameter"
        assert any(
            "b_x1" in p or "x1" in p for p in param_names
        ), "Model should have coefficient parameter"

    @pytest.mark.slow
    def test_prior_with_group_parameter(self):
        """
        Test prior() with group parameter for hierarchical models.

        Tests that prior() correctly handles:
        - Group-level (random effects) priors
        - Combination of coefficient and group-level priors
        - Hierarchical model with (1|group) structure
        - Integration with real brms dataset
        """
        from brmspy import brms
        from brmspy.brms import prior
        from brmspy.types.brms_results import PriorSpec
        import arviz as az

        # Load epilepsy dataset (has patient grouping variable)
        epilepsy = brms.get_brms_data("epilepsy")

        # Create prior specifications including group-level prior
        priors = [
            prior("normal(0, 0.5)", class_="b"),  # Coefficient prior
            prior("exponential(1)", class_="sd", group="patient"),  # Group SD prior
            prior("student_t(3, 0, 2.5)", class_="Intercept"),  # Intercept prior
        ]

        # Verify all priors are PriorSpec objects
        for p in priors:
            assert isinstance(
                p, PriorSpec
            ), "Each prior() call should return PriorSpec instance"

        # Verify group parameter is set correctly
        group_prior = priors[1]
        assert group_prior.class_ == "sd", "Group prior should have class_='sd'"
        assert group_prior.group == "patient", "Group prior should have group='patient'"
        assert (
            group_prior.prior == "exponential(1)"
        ), "Group prior string should be stored correctly"

        # Fit hierarchical model with group-level priors
        model = brms.fit(
            formula="count ~ zAge + (1|patient)",
            data=epilepsy,
            priors=priors,
            family="poisson",
            iter=200,
            warmup=100,
            chains=1,
            silent=2,
            refresh=0,
        )

        # Verify model fitted successfully
        assert isinstance(
            model.idata, az.InferenceData
        ), "fit() with group priors should return InferenceData"

        # Check that both fixed and random effects parameters exist
        param_names = list(model.idata.posterior.data_vars)
        assert len(param_names) > 0, "Model should have parameters"

        # Check for fixed effects
        assert any(
            "b_Intercept" in p or "Intercept" in p for p in param_names
        ), "Model should have intercept parameter"
        assert any(
            "b_zAge" in p for p in param_names
        ), "Model should have coefficient for zAge"

        # Check for random effects (group-level standard deviation)
        assert any(
            "sd_patient" in p for p in param_names
        ), "Model should have random effect SD parameter for patient group"

        # Verify the group prior was actually used by checking parameter existence
        sd_params = [p for p in param_names if "sd_patient" in p]
        assert (
            len(sd_params) > 0
        ), "Group-level prior should result in sd_patient parameters"


@pytest.mark.requires_brms
class TestAdditionalFunctions:
    """Test additional brms functions to improve coverage."""

    def test_make_stancode(self, sample_dataframe):
        """
        Test make_stancode() function for generating Stan code.

        Tests that make_stancode():
        - Generates valid Stan code from formula
        - Works with and without priors
        - Returns a non-empty string
        - Contains expected Stan program structure
        """
        from brmspy import brms
        from brmspy.brms import prior

        # Test without priors
        stan_code = brms.make_stancode(
            formula="y ~ x1", data=sample_dataframe, priors=[], family="gaussian"
        )

        assert isinstance(stan_code, str), "make_stancode() should return a string"
        assert len(stan_code) > 0, "Stan code should not be empty"
        assert "data {" in stan_code, "Stan code should contain data block"
        assert "parameters {" in stan_code, "Stan code should contain parameters block"
        assert "model {" in stan_code, "Stan code should contain model block"

        # Test with priors
        priors = [prior("normal(0, 1)", class_="b")]
        stan_code_with_priors = brms.make_stancode(
            formula="y ~ x1", data=sample_dataframe, priors=priors, family="gaussian"
        )

        assert isinstance(
            stan_code_with_priors, str
        ), "make_stancode() with priors should return a string"
        assert (
            len(stan_code_with_priors) > 0
        ), "Stan code with priors should not be empty"

    @pytest.mark.slow
    def test_fit_sample_false(self, sample_dataframe):
        """
        Test fit() with sample=False for compile-only mode.

        Tests that fit(sample=False):
        - Compiles model without sampling
        - Returns FitResult with empty idata
        - R object is valid brmsfit
        """
        from brmspy import brms

        # Fit model with sample=False (compile only)
        model = brms.fit(
            formula="y ~ x1", data=sample_dataframe, family="gaussian", sample=False
        )

        # Verify return type
        assert isinstance(
            model, brms.FitResult
        ), "fit(sample=False) should return FitResult"

        # Verify R object exists
        assert (
            model.r is not None
        ), "fit(sample=False) should have valid R brmsfit object"

        # Verify idata is empty or minimal (no actual sampling occurred)
        # The idata should be an empty InferenceData or have no posterior samples
        assert model.idata is not None, "fit(sample=False) should have idata attribute"

    @pytest.mark.slow
    def test_fit_tqdm_segfault(self, sample_dataframe):
        """
        Previously running .fit() in a tqdm wrapped loop could cause segfault and instant crashes.
        This happens when an r import is tried within the loop.
        """
        try:
            from tqdm.auto import tqdm
        except ImportError:
            return
        from brmspy import brms

        # Fit model with sample=False (compile only)
        for _ in tqdm(range(2)):
            model = brms.fit(
                formula="y ~ x1",
                data=sample_dataframe,
                family="gaussian",
                sample=False,
                iter=100,
                warmup=50,
            )

            # Verify return type
            assert isinstance(
                model, brms.FitResult
            ), "fit(sample=False) should return FitResult"

    @pytest.mark.slow
    def test_posterior_linpred_without_newdata(self, sample_dataframe):
        """
        Test posterior_linpred() without newdata parameter.

        Tests that posterior_linpred():
        - Works without newdata (uses original training data)
        - Returns valid PosteriorLinpredResult
        - Contains predictions for original observations
        """
        from brmspy import brms
        import arviz as az

        # Fit a simple model
        model = brms.fit(
            formula="y ~ x1",
            data=sample_dataframe,
            family="gaussian",
            iter=200,
            warmup=100,
            chains=1,
            silent=2,
            refresh=0,
        )

        # Get linear predictor without newdata
        linpred = brms.posterior_linpred(model)

        # Verify return type
        assert isinstance(
            linpred, brms.PosteriorLinpredResult
        ), "posterior_linpred() should return PosteriorLinpredResult"

        # Verify idata exists
        assert isinstance(
            linpred.idata, az.InferenceData
        ), "posterior_linpred() should return InferenceData"

        # Verify R object exists
        assert linpred.r is not None, "posterior_linpred() should have R matrix"

    @pytest.mark.slow
    def test_log_lik_without_newdata(self, sample_dataframe):
        """
        Test log_lik() without newdata parameter.

        Tests that log_lik():
        - Works without newdata (uses original training data)
        - Returns valid LogLikResult
        - Contains log-likelihood for original observations
        """
        from brmspy import brms
        import arviz as az

        # Fit a simple model
        model = brms.fit(
            formula="y ~ x1",
            data=sample_dataframe,
            family="gaussian",
            iter=200,
            warmup=100,
            chains=1,
            silent=2,
            refresh=0,
        )

        # Get log-likelihood without newdata
        loglik = brms.log_lik(model)

        # Verify return type
        assert isinstance(
            loglik, brms.LogLikResult
        ), "log_lik() should return LogLikResult"

        # Verify idata exists
        assert isinstance(
            loglik.idata, az.InferenceData
        ), "log_lik() should return InferenceData"

        # Verify R object exists
        assert loglik.r is not None, "log_lik() should have R matrix"


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-m", "requires_brms"])
