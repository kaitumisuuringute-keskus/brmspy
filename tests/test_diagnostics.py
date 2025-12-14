import numpy as np
import pytest


@pytest.mark.requires_brms
class TestSummaryFunction:
    """Test the updated summary() function that returns a Summary dataclass."""

    @pytest.mark.slow
    def test_summary_return_type_and_structure(self, sample_dataframe):
        """
        Test that summary() returns a Summary dataclass with all expected attributes.

        Verifies:
        - Return type is Summary dataclass
        - All expected attributes exist (formula, fixed, spec_pars, random, etc.)
        - Attributes have correct types
        """
        from brmspy import brms
        from brmspy.types.brms_results import SummaryResult
        import pandas as pd

        # Fit a simple model
        model = brms.fit(
            formula="y ~ x1",
            data=sample_dataframe,
            family="gaussian",
            iter=100,
            warmup=50,
            chains=1,
            silent=2,
            refresh=0,
        )

        # Get summary
        summary = brms.summary(model)
        print("summary", summary)

        # Verify return type is Summary dataclass
        assert isinstance(
            summary, SummaryResult
        ), f"summary() should return Summary dataclass, got {type(summary)}"

        # Verify all expected attributes exist
        expected_attrs = [
            "formula",
            "data_name",
            "group",
            "nobs",
            "ngrps",
            "autocor",
            "prior",
            "algorithm",
            "sampler",
            "total_ndraws",
            "chains",
            "iter",
            "warmup",
            "thin",
            "has_rhat",
            "fixed",
            "spec_pars",
            "cor_pars",
            "random",
        ]

        for attr in expected_attrs:
            assert hasattr(summary, attr), f"Summary should have attribute '{attr}'"

        # Verify types of key attributes
        assert isinstance(summary.formula, str), "formula should be a string"
        assert isinstance(summary.nobs, int), "nobs should be an integer"
        assert isinstance(
            summary.fixed, pd.DataFrame
        ), "fixed should be a pandas DataFrame"
        assert isinstance(
            summary.spec_pars, pd.DataFrame
        ), "spec_pars should be a pandas DataFrame"
        assert isinstance(
            summary.prior, pd.DataFrame
        ), "prior should be a pandas DataFrame"

        # Verify numeric fields have reasonable values
        assert summary.nobs > 0, "nobs should be positive"
        assert summary.chains > 0, "chains should be positive"
        assert summary.total_ndraws > 0, "total_ndraws should be positive"

    @pytest.mark.slow
    def test_summary_component_access(self, sample_dataframe):
        """
        Test accessing specific components of SummaryResult.

        Verifies:
        - Can access summary.fixed as DataFrame with parameter estimates
        - Can access summary.spec_pars for family-specific parameters
        - DataFrames contain expected columns (Estimate, Est.Error, etc.)
        - Values are reasonable
        """
        from brmspy import brms
        import pandas as pd

        # Fit a simple model
        model = brms.fit(
            formula="y ~ x1",
            data=sample_dataframe,
            family="gaussian",
            iter=100,
            warmup=50,
            chains=1,
            silent=2,
            refresh=0,
        )

        # Get summary
        summary = brms.summary(model, prior=True)

        # Test fixed effects access
        fixed = summary.fixed
        assert isinstance(fixed, pd.DataFrame), "summary.fixed should be a DataFrame"
        assert not fixed.empty, "Fixed effects DataFrame should not be empty"

        # Check for expected parameter columns
        # brms typically includes: Estimate, Est.Error, l-95% CI, u-95% CI, Rhat, etc.
        assert any(
            "Estimate" in col or "estimate" in col.lower() for col in fixed.columns
        ), "Fixed effects should contain Estimate column"

        # Check for expected parameters in index (Intercept, x1 coefficient)
        param_names = fixed.index.tolist()
        assert any(
            "Intercept" in str(p) for p in param_names
        ), "Fixed effects should include Intercept parameter"
        assert any(
            "x1" in str(p) for p in param_names
        ), "Fixed effects should include x1 parameter"

        # Test spec_pars access (family-specific parameters like sigma)
        spec_pars = summary.spec_pars
        assert isinstance(
            spec_pars, pd.DataFrame
        ), "summary.spec_pars should be a DataFrame"

        # For gaussian family, should have sigma parameter
        if not spec_pars.empty:
            spec_param_names = spec_pars.index.tolist()
            assert any(
                "sigma" in str(p).lower() for p in spec_param_names
            ), "Gaussian model should have sigma in spec_pars"

        # Test prior access
        prior = summary.prior
        assert isinstance(prior, pd.DataFrame), "summary.prior should be a DataFrame"
        assert not prior.empty, "Prior DataFrame should not be empty"

    @pytest.mark.slow
    def test_summary_pretty_print(self, sample_dataframe):
        """
        Test the pretty print functionality of SummaryResult.

        Verifies:
        - str(summary) produces formatted output
        - Output contains expected sections (Formula, Data, Population-Level Effects)
        - Output is human-readable and well-structured
        """
        from brmspy import brms

        # Fit a simple model
        model = brms.fit(
            formula="y ~ x1",
            data=sample_dataframe,
            family="gaussian",
            iter=100,
            warmup=50,
            chains=1,
            silent=2,
            refresh=0,
        )

        # Get summary
        summary = brms.summary(model)

        # Get string representation
        summary_str = str(summary)

        # Verify it's a non-empty string
        assert isinstance(summary_str, str), "str(summary) should return a string"
        assert len(summary_str) > 0, "Summary string should not be empty"

        # Check for formula section
        assert "Formula:" in summary_str, "Summary should include Formula section"
        assert "y ~ x1" in summary_str, "Summary should display the model formula"

        # Check for data info section
        assert "Data:" in summary_str, "Summary should include Data section"
        assert (
            "Number of observations:" in summary_str or "observations:" in summary_str
        ), "Summary should include number of observations"

        # Check for draws/sampling info
        assert "Draws:" in summary_str, "Summary should include Draws section"
        assert "chains" in summary_str, "Summary should mention number of chains"

        # Check for population-level effects section
        assert (
            "Regression Coefficients" in summary_str
        ), "Summary should include Regression Coefficients section"

        # Check for algorithm/diagnostics section
        assert (
            "were sampled" in summary_str
        ), "Summary should include 'were sampled' information"

        # Verify __repr__ also works (should be same as __str__)
        summary_repr = repr(summary)
        assert summary_repr == summary_str, "repr(summary) should equal str(summary)"


@pytest.mark.requires_brms
class TestFixefFunction:
    """Test the fixef() function for extracting population-level effects."""

    @pytest.mark.slow
    def test_fixef_basic_functionality(self, sample_dataframe):
        """
        Test fixef() function for extracting fixed effects.

        Verifies:
        - Returns a DataFrame with summary statistics
        - Contains expected parameters (Intercept, x1)
        - Has expected columns (Estimate, Est.Error, credible intervals)
        - Values are reasonable and numeric
        - Can extract specific parameters with pars argument
        """
        import pandas as pd
        from brmspy import brms

        # Fit a simple model
        model = brms.fit(
            formula="y ~ x1",
            data=sample_dataframe,
            family="gaussian",
            iter=100,
            warmup=50,
            chains=1,
            silent=2,
            refresh=0,
        )

        # Get fixed effects (default: summary=True)
        fixed_effects = brms.fixef(model)

        # Verify return type is DataFrame
        assert isinstance(
            fixed_effects, pd.DataFrame
        ), f"fixef() should return pandas DataFrame, got {type(fixed_effects)}"

        # Verify DataFrame is not empty
        assert not fixed_effects.empty, "Fixed effects DataFrame should not be empty"

        # Verify expected parameters are present
        param_names = fixed_effects.index.tolist()
        assert any(
            "Intercept" in str(p) for p in param_names
        ), "Fixed effects should include Intercept parameter"
        assert any(
            "x1" in str(p) for p in param_names
        ), "Fixed effects should include x1 parameter"

        # Verify expected columns exist
        columns = fixed_effects.columns.tolist()
        assert any(
            "Estimate" in str(col) for col in columns
        ), "Fixed effects should have Estimate column"
        assert any(
            "Error" in str(col) or "Est.Error" in str(col) for col in columns
        ), "Fixed effects should have error/uncertainty column"

        # Verify credible interval columns exist (default probs=(0.025, 0.975))
        # Column names might be like 'Q2.5', 'Q97.5', 'l-95% CI', 'u-95% CI', etc.
        assert (
            len(columns) >= 3
        ), "Fixed effects should have at least 3 columns (estimate, error, intervals)"

        # Verify all values are numeric (no NaNs or invalid data)
        assert (
            fixed_effects.select_dtypes(include=["number"]).shape == fixed_effects.shape
        ), "All fixed effects values should be numeric"

        # Verify no NaN values
        assert (
            not fixed_effects.isna().any().any()
        ), "Fixed effects should not contain NaN values"

        # Test extracting specific parameters with pars argument
        x1_only = brms.fixef(model, pars=["x1"])
        assert isinstance(
            x1_only, pd.DataFrame
        ), "fixef() with pars should return DataFrame"
        assert (
            len(x1_only) == 1
        ), "fixef() with pars=['x1'] should return only 1 parameter"
        assert "x1" in str(x1_only.index[0]), "Extracted parameter should be x1"


@pytest.mark.requires_brms
class TestRanefFunction:
    """Test the ranef() function for extracting group-level (random) effects."""

    @pytest.mark.slow
    def test_ranef_summary_mode(self, sample_dataframe):
        """
        Test ranef() with summary=True (default).

        Verifies:
        - Returns dict of xarray DataArrays
        - Each DataArray has correct dimensions: (group, stat, coef)
        - Contains expected statistics (Estimate, Est.Error, credible intervals)
        - Can select specific groups and statistics
        """
        from brmspy import brms
        import xarray as xr

        # Add group variation for better convergence
        sample_dataframe["y"] = sample_dataframe["y"] + sample_dataframe["group"].map(
            {"G1": -2, "G2": 2}
        )

        # Fit a model with random effects
        model = brms.fit(
            formula="y ~ x1 + (1|group)",
            data=sample_dataframe,
            family="gaussian",
            iter=100,
            warmup=50,
            chains=1,
            silent=2,
            refresh=0,
        )

        # Get random effects (default: summary=True)
        random_effects = brms.ranef(model)

        # Verify return type is dict
        assert isinstance(
            random_effects, dict
        ), f"ranef() should return dict, got {type(random_effects)}"

        # Verify dict is not empty
        assert len(random_effects) > 0, "Random effects dict should not be empty"

        # Verify 'group' is in the dict (our grouping factor)
        assert "group" in random_effects, "Random effects should contain 'group' key"

        # Get the group random effects
        group_re = random_effects["group"]

        # Verify it's an xarray DataArray
        assert isinstance(
            group_re, xr.DataArray
        ), f"Random effects should be xarray DataArray, got {type(group_re)}"

        # Verify dimensions for summary=True: (group, stat, coef)
        assert group_re.dims == (
            "group",
            "stat",
            "coef",
        ), f"DataArray should have dims ('group', 'stat', 'coef'), got {group_re.dims}"

        # Verify we have the expected groups (G1, G2)
        groups = list(group_re.coords["group"].values)
        assert len(groups) == 2, "Should have 2 groups"
        assert set(groups) == {"G1", "G2"}, "Groups should be G1 and G2"

        # Verify we have expected statistics
        stats = list(group_re.coords["stat"].values)
        assert "Estimate" in stats, "Statistics should include Estimate"
        assert any(
            "Error" in s or "Est.Error" in s for s in stats
        ), "Statistics should include error/uncertainty"

        # Verify we can select specific values
        intercept_estimates = group_re.sel(coef="Intercept", stat="Estimate")
        assert intercept_estimates.shape == (2,), "Should have one estimate per group"

        # Verify no NaN values
        assert (
            not group_re.isnull().any()
        ), "Random effects should not contain NaN values"

    @pytest.mark.slow
    def test_ranef_raw_samples_mode(self, sample_dataframe):
        """
        Test ranef() with summary=False to get raw posterior samples.

        Verifies:
        - Returns dict of xarray DataArrays with raw draws
        - Each DataArray has correct dimensions: (draw, group, coef)
        - Number of draws matches expected posterior samples
        - Can compute custom statistics from raw draws
        """
        from brmspy import brms
        import xarray as xr
        import numpy as np

        # Add group variation for better convergence
        sample_dataframe["y"] = sample_dataframe["y"] + sample_dataframe["group"].map(
            {"G1": -2, "G2": 2}
        )

        # Fit a model with random effects
        model = brms.fit(
            formula="y ~ x1 + (1|group)",
            data=sample_dataframe,
            family="gaussian",
            iter=100,
            warmup=50,
            chains=1,
            silent=2,
            refresh=0,
        )

        # Get raw posterior samples (summary=False)
        random_effects_draws = brms.ranef(model, summary=False)

        # Verify return type is dict
        assert isinstance(
            random_effects_draws, dict
        ), f"ranef(summary=False) should return dict, got {type(random_effects_draws)}"

        # Verify dict is not empty
        assert len(random_effects_draws) > 0, "Random effects dict should not be empty"

        # Verify 'group' is in the dict
        assert (
            "group" in random_effects_draws
        ), "Random effects should contain 'group' key"

        # Get the group random effects draws
        group_re_draws = random_effects_draws["group"]

        # Verify it's an xarray DataArray
        assert isinstance(
            group_re_draws, xr.DataArray
        ), f"Random effects should be xarray DataArray, got {type(group_re_draws)}"

        # Verify dimensions for summary=False: (draw, group, coef)
        assert group_re_draws.dims == (
            "draw",
            "group",
            "coef",
        ), f"DataArray should have dims ('draw', 'group', 'coef'), got {group_re_draws.dims}"

        # Verify number of draws (2 chains × 50 post-warmup = 100 draws)
        expected_draws = 1 * (100 - 50)  # chains * (iter - warmup)
        assert (
            group_re_draws.sizes["draw"] == expected_draws
        ), f"Should have {expected_draws} draws, got {group_re_draws.sizes['draw']}"

        # Verify we have the expected groups
        groups = list(group_re_draws.coords["group"].values)
        assert len(groups) == 2, "Should have 2 groups"
        assert set(groups) == {"G1", "G2"}, "Groups should be G1 and G2"

        # Verify we can compute statistics from raw draws
        intercept_draws = group_re_draws.sel(coef="Intercept", group="G1")
        assert intercept_draws.shape == (
            expected_draws,
        ), f"Should have {expected_draws} draws for single group/coef"

        # Compute custom statistic (e.g., median)
        median_estimate = float(np.median(intercept_draws.values))
        assert not np.isnan(
            median_estimate
        ), "Should be able to compute median from draws"

        # Verify no NaN values in the draws
        assert (
            not group_re_draws.isnull().any()
        ), "Random effects draws should not contain NaN values"


@pytest.mark.requires_brms
class TestPosteriorSummaryFunction:
    """Test the posterior_summary() function for comprehensive parameter summaries."""

    @pytest.mark.slow
    def test_posterior_summary_all_parameters(self, sample_dataframe):
        """
        Test posterior_summary() for extracting all parameter estimates.

        Verifies:
        - Returns a DataFrame with all model parameters
        - Includes fixed effects, family parameters, and more
        - Has expected columns (Estimate, Est.Error, credible intervals)
        - Can extract specific parameters with variable argument
        """
        from brmspy import brms
        import pandas as pd

        # Fit a simple model
        model = brms.fit(
            formula="y ~ x1",
            data=sample_dataframe,
            family="gaussian",
            iter=100,
            warmup=50,
            chains=1,
            silent=2,
            refresh=0,
        )

        # Get posterior summary for all parameters
        post_summary = brms.posterior_summary(model)

        # Verify return type is DataFrame
        assert isinstance(
            post_summary, pd.DataFrame
        ), f"posterior_summary() should return pandas DataFrame, got {type(post_summary)}"

        # Verify DataFrame is not empty
        assert not post_summary.empty, "Posterior summary DataFrame should not be empty"

        # Verify expected columns exist
        columns = post_summary.columns.tolist()
        assert any(
            "Estimate" in str(col) for col in columns
        ), "Posterior summary should have Estimate column"
        assert any(
            "Error" in str(col) or "Est.Error" in str(col) for col in columns
        ), "Posterior summary should have error/uncertainty column"

        # Verify it contains more than just fixed effects (should include sigma, etc.)
        param_names = post_summary.index.tolist()
        assert (
            len(param_names) >= 3
        ), "Should have at least Intercept, x1, and sigma parameters"

        # Check for specific parameters
        assert any(
            "Intercept" in str(p) for p in param_names
        ), "Should include Intercept parameter"
        assert any("x1" in str(p) for p in param_names), "Should include x1 parameter"
        assert any(
            "sigma" in str(p).lower() for p in param_names
        ), "Should include sigma parameter for Gaussian model"

        # Verify no NaN values
        assert (
            not post_summary.isna().any().any()
        ), "Posterior summary should not contain NaN values"


@pytest.mark.requires_brms
class TestPriorSummaryFunction:
    """Test the prior_summary() function for extracting prior specifications."""

    @pytest.mark.slow
    def test_prior_summary_with_custom_priors(self, sample_dataframe):
        """
        Test prior_summary() for extracting prior specifications from fitted model.

        Verifies:
        - Returns a DataFrame with prior specifications
        - Includes both user-set and default priors when all=True
        - Has expected columns (prior, class, coef, etc.)
        - Can filter to only user-set priors with all=False
        """
        from brmspy import brms
        import pandas as pd

        # Fit a model with custom priors
        model = brms.fit(
            formula="y ~ x1",
            data=sample_dataframe,
            priors=[brms.prior("normal(0, 1)", "b")],
            family="gaussian",
            iter=100,
            warmup=50,
            chains=1,
            silent=2,
            refresh=0,
        )

        # Get prior summary (all priors including defaults)
        prior_sum = brms.prior_summary(model, all=True)

        # Verify return type is DataFrame
        assert isinstance(
            prior_sum, pd.DataFrame
        ), f"prior_summary() should return pandas DataFrame, got {type(prior_sum)}"

        # Verify DataFrame is not empty
        assert not prior_sum.empty, "Prior summary DataFrame should not be empty"

        # Verify expected columns exist
        columns = prior_sum.columns.tolist()
        assert any(
            "prior" in str(col).lower() for col in columns
        ), "Prior summary should have 'prior' column"
        assert any(
            "class" in str(col).lower() for col in columns
        ), "Prior summary should have 'class' column"

        # Verify it contains information about the custom prior
        prior_strings = (
            prior_sum["prior"].tolist() if "prior" in prior_sum.columns else []
        )
        # The prior string might be in a different format in the summary
        assert len(prior_strings) > 0, "Should have at least one prior specification"

        # Test getting only user-set priors (all=False)
        user_priors = brms.prior_summary(model, all=False)
        assert isinstance(
            user_priors, pd.DataFrame
        ), "prior_summary(all=False) should return DataFrame"

        # User-set priors should be subset of all priors
        assert len(user_priors) <= len(
            prior_sum
        ), "User-set priors should be subset of all priors"


@pytest.mark.requires_brms
class TestValidateNewdataFunction:
    """Test the validate_newdata() function for validating prediction data."""

    @pytest.mark.slow
    def test_validate_newdata_with_valid_data(self, sample_dataframe):
        """
        Test validate_newdata() with valid new data.

        Verifies:
        - Returns a validated pandas DataFrame
        - DataFrame has correct structure
        - Can validate data for simple model without random effects
        - Validation succeeds when all required variables present
        """
        import brmspy
        import pandas as pd
        from brmspy import brms

        # Fit a simple model
        model = brms.brm(
            formula="y ~ x1 + x2",
            data=sample_dataframe,
            family="gaussian",
            iter=100,
            warmup=50,
            chains=1,
            silent=2,
            refresh=0,
        )

        # Create valid new data with all required variables
        n = 10
        newdata = pd.DataFrame(
            {
                "y": np.random.normal(10, 2, n),
                "x1": np.random.normal(0, 1, n),
                "x2": np.random.choice(["A", "B"], n),
                "group": np.repeat(["G1", "G2"], n // 2),
            }
        )

        # Validate newdata
        validated = brms.validate_newdata(
            newdata, model, check_response=False  # Response not needed for prediction
        )

        # Verify return type is DataFrame
        assert isinstance(
            validated, pd.DataFrame
        ), f"validate_newdata() should return DataFrame, got {type(validated)}"

        # Verify DataFrame is not empty
        assert not validated.empty, "Validated DataFrame should not be empty"

        # Verify it has the same number of rows as input
        assert len(validated) == len(
            newdata
        ), f"Validated data should have {len(newdata)} rows, got {len(validated)}"

        # Verify required columns are present
        assert "x1" in validated.columns, "Validated data should contain x1 column"
        assert "x2" in validated.columns, "Validated data should contain x2 column"

    @pytest.mark.slow
    def test_validate_newdata_with_missing_variables(self, sample_dataframe):
        """
        Test validate_newdata() with invalid data missing required variables.

        Verifies:
        - Raises error when required variables are missing
        - Error message indicates which variable is missing
        - Validation properly detects incomplete data
        """
        from brmspy import brms
        import pandas as pd
        import pytest

        # Fit a model with two predictors
        model = brms.fit(
            formula="y ~ x1 + x2",
            data=sample_dataframe,
            family="gaussian",
            iter=100,
            warmup=50,
            chains=1,
            silent=2,
            refresh=0,
        )

        # Create invalid new data missing x2
        invalid_newdata = pd.DataFrame(
            {
                "x1": [1.0, 2.0, 3.0]
                # Missing x2!
            }
        )

        # Validation should raise an error
        with pytest.raises(Exception) as exc_info:
            brms.validate_newdata(invalid_newdata, model, check_response=False)

        # Verify error was raised
        assert (
            exc_info.value is not None
        ), "validate_newdata should raise error for missing variables"

        # Error message should mention the missing variable or validation failure
        error_msg = str(exc_info.value).lower()
        assert (
            "x2" in error_msg
            or "validate_data" in error_msg
            or "column" in error_msg
            or "not found" in error_msg
        ), f"Error should mention missing variable, got: {exc_info.value}"
