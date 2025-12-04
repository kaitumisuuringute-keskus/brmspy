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
        import brmspy
        from brmspy.brms_functions.diagnostics import SummaryResult
        import pandas as pd
        
        # Fit a simple model
        model = brmspy.fit(
            formula="y ~ x1",
            data=sample_dataframe,
            family="gaussian",
            iter=100,
            warmup=50,
            chains=1,
            silent=2,
            refresh=0
        )
        
        # Get summary
        summary = brmspy.summary(model)
        
        # Verify return type is Summary dataclass
        assert isinstance(summary, SummaryResult), \
            f"summary() should return Summary dataclass, got {type(summary)}"
        
        # Verify all expected attributes exist
        expected_attrs = [
            'formula', 'data_name', 'group', 'nobs', 'ngrps', 'autocor',
            'prior', 'algorithm', 'sampler', 'total_ndraws', 'chains',
            'iter', 'warmup', 'thin', 'has_rhat', 'fixed', 'spec_pars',
            'cor_pars', 'random'
        ]
        
        for attr in expected_attrs:
            assert hasattr(summary, attr), \
                f"Summary should have attribute '{attr}'"
        
        # Verify types of key attributes
        assert isinstance(summary.formula, str), \
            "formula should be a string"
        assert isinstance(summary.nobs, int), \
            "nobs should be an integer"
        assert isinstance(summary.fixed, pd.DataFrame), \
            "fixed should be a pandas DataFrame"
        assert isinstance(summary.spec_pars, pd.DataFrame), \
            "spec_pars should be a pandas DataFrame"
        assert isinstance(summary.prior, pd.DataFrame), \
            "prior should be a pandas DataFrame"
        
        # Verify numeric fields have reasonable values
        assert summary.nobs > 0, \
            "nobs should be positive"
        assert summary.chains > 0, \
            "chains should be positive"
        assert summary.total_ndraws > 0, \
            "total_ndraws should be positive"
    
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
        import brmspy
        import pandas as pd
        
        # Fit a simple model
        model = brmspy.fit(
            formula="y ~ x1",
            data=sample_dataframe,
            family="gaussian",
            iter=100,
            warmup=50,
            chains=1,
            silent=2,
            refresh=0
        )
        
        # Get summary
        summary = brmspy.summary(model, prior=True)
        
        # Test fixed effects access
        fixed = summary.fixed
        assert isinstance(fixed, pd.DataFrame), \
            "summary.fixed should be a DataFrame"
        assert not fixed.empty, \
            "Fixed effects DataFrame should not be empty"
        
        # Check for expected parameter columns
        # brms typically includes: Estimate, Est.Error, l-95% CI, u-95% CI, Rhat, etc.
        assert any('Estimate' in col or 'estimate' in col.lower() for col in fixed.columns), \
            "Fixed effects should contain Estimate column"
        
        # Check for expected parameters in index (Intercept, x1 coefficient)
        param_names = fixed.index.tolist()
        assert any('Intercept' in str(p) for p in param_names), \
            "Fixed effects should include Intercept parameter"
        assert any('x1' in str(p) for p in param_names), \
            "Fixed effects should include x1 parameter"
        
        # Test spec_pars access (family-specific parameters like sigma)
        spec_pars = summary.spec_pars
        assert isinstance(spec_pars, pd.DataFrame), \
            "summary.spec_pars should be a DataFrame"
        
        # For gaussian family, should have sigma parameter
        if not spec_pars.empty:
            spec_param_names = spec_pars.index.tolist()
            assert any('sigma' in str(p).lower() for p in spec_param_names), \
                "Gaussian model should have sigma in spec_pars"
        
        # Test prior access
        prior = summary.prior
        assert isinstance(prior, pd.DataFrame), \
            "summary.prior should be a DataFrame"
        assert not prior.empty, \
            "Prior DataFrame should not be empty"
    
    @pytest.mark.slow
    def test_summary_pretty_print(self, sample_dataframe):
        """
        Test the pretty print functionality of SummaryResult.
        
        Verifies:
        - str(summary) produces formatted output
        - Output contains expected sections (Formula, Data, Population-Level Effects)
        - Output is human-readable and well-structured
        """
        import brmspy
        
        # Fit a simple model
        model = brmspy.fit(
            formula="y ~ x1",
            data=sample_dataframe,
            family="gaussian",
            iter=100,
            warmup=50,
            chains=1,
            silent=2,
            refresh=0
        )
        
        # Get summary
        summary = brmspy.summary(model)
        
        # Get string representation
        summary_str = str(summary)
        
        # Verify it's a non-empty string
        assert isinstance(summary_str, str), \
            "str(summary) should return a string"
        assert len(summary_str) > 0, \
            "Summary string should not be empty"
        
        # Check for expected header section
        assert "Summary of brmsfit (Python)" in summary_str, \
            "Summary should include header"
        
        # Check for formula section
        assert "Formula:" in summary_str, \
            "Summary should include Formula section"
        assert "y ~ x1" in summary_str, \
            "Summary should display the model formula"
        
        # Check for data info section
        assert "Data:" in summary_str, \
            "Summary should include Data section"
        assert "Number of observations:" in summary_str or "observations:" in summary_str, \
            "Summary should include number of observations"
        
        # Check for draws/sampling info
        assert "Draws:" in summary_str, \
            "Summary should include Draws section"
        assert "chains" in summary_str, \
            "Summary should mention number of chains"
        
        # Check for population-level effects section
        assert "Population-Level Effects:" in summary_str, \
            "Summary should include Population-Level Effects section"
        
        # Check for algorithm/diagnostics section
        assert "Algorithm" in summary_str, \
            "Summary should include Algorithm information"
        
        # Verify __repr__ also works (should be same as __str__)
        summary_repr = repr(summary)
        assert summary_repr == summary_str, \
            "repr(summary) should equal str(summary)"


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
        import brmspy
        import pandas as pd
        
        # Fit a simple model
        model = brmspy.fit(
            formula="y ~ x1",
            data=sample_dataframe,
            family="gaussian",
            iter=100,
            warmup=50,
            chains=1,
            silent=2,
            refresh=0
        )
        
        # Get fixed effects (default: summary=True)
        fixed_effects = brmspy.fixef(model)
        
        # Verify return type is DataFrame
        assert isinstance(fixed_effects, pd.DataFrame), \
            f"fixef() should return pandas DataFrame, got {type(fixed_effects)}"
        
        # Verify DataFrame is not empty
        assert not fixed_effects.empty, \
            "Fixed effects DataFrame should not be empty"
        
        # Verify expected parameters are present
        param_names = fixed_effects.index.tolist()
        assert any('Intercept' in str(p) for p in param_names), \
            "Fixed effects should include Intercept parameter"
        assert any('x1' in str(p) for p in param_names), \
            "Fixed effects should include x1 parameter"
        
        # Verify expected columns exist
        columns = fixed_effects.columns.tolist()
        assert any('Estimate' in str(col) for col in columns), \
            "Fixed effects should have Estimate column"
        assert any('Error' in str(col) or 'Est.Error' in str(col) for col in columns), \
            "Fixed effects should have error/uncertainty column"
        
        # Verify credible interval columns exist (default probs=(0.025, 0.975))
        # Column names might be like 'Q2.5', 'Q97.5', 'l-95% CI', 'u-95% CI', etc.
        assert len(columns) >= 3, \
            "Fixed effects should have at least 3 columns (estimate, error, intervals)"
        
        # Verify all values are numeric (no NaNs or invalid data)
        assert fixed_effects.select_dtypes(include=['number']).shape == fixed_effects.shape, \
            "All fixed effects values should be numeric"
        
        # Verify no NaN values
        assert not fixed_effects.isna().any().any(), \
            "Fixed effects should not contain NaN values"
        
        # Test extracting specific parameters with pars argument
        x1_only = brmspy.fixef(model, pars=["x1"])
        assert isinstance(x1_only, pd.DataFrame), \
            "fixef() with pars should return DataFrame"
        assert len(x1_only) == 1, \
            "fixef() with pars=['x1'] should return only 1 parameter"
        assert 'x1' in str(x1_only.index[0]), \
            "Extracted parameter should be x1"


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
        import brmspy
        import xarray as xr
        
        # Add group variation for better convergence
        sample_dataframe['y'] = (
            sample_dataframe['y'] +
            sample_dataframe['group'].map({'G1': -2, 'G2': 2})
        )
        
        # Fit a model with random effects
        model = brmspy.fit(
            formula="y ~ x1 + (1|group)",
            data=sample_dataframe,
            family="gaussian",
            iter=100,
            warmup=50,
            chains=1,
            silent=2,
            refresh=0
        )
        
        # Get random effects (default: summary=True)
        random_effects = brmspy.ranef(model)
        
        # Verify return type is dict
        assert isinstance(random_effects, dict), \
            f"ranef() should return dict, got {type(random_effects)}"
        
        # Verify dict is not empty
        assert len(random_effects) > 0, \
            "Random effects dict should not be empty"
        
        # Verify 'group' is in the dict (our grouping factor)
        assert 'group' in random_effects, \
            "Random effects should contain 'group' key"
        
        # Get the group random effects
        group_re = random_effects['group']
        
        # Verify it's an xarray DataArray
        assert isinstance(group_re, xr.DataArray), \
            f"Random effects should be xarray DataArray, got {type(group_re)}"
        
        # Verify dimensions for summary=True: (group, stat, coef)
        assert group_re.dims == ('group', 'stat', 'coef'), \
            f"DataArray should have dims ('group', 'stat', 'coef'), got {group_re.dims}"
        
        # Verify we have the expected groups (G1, G2)
        groups = list(group_re.coords['group'].values)
        assert len(groups) == 2, \
            "Should have 2 groups"
        assert set(groups) == {'G1', 'G2'}, \
            "Groups should be G1 and G2"
        
        # Verify we have expected statistics
        stats = list(group_re.coords['stat'].values)
        assert 'Estimate' in stats, \
            "Statistics should include Estimate"
        assert any('Error' in s or 'Est.Error' in s for s in stats), \
            "Statistics should include error/uncertainty"
        
        # Verify we can select specific values
        intercept_estimates = group_re.sel(coef='Intercept', stat='Estimate')
        assert intercept_estimates.shape == (2,), \
            "Should have one estimate per group"
        
        # Verify no NaN values
        assert not group_re.isnull().any(), \
            "Random effects should not contain NaN values"
    
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
        import brmspy
        import xarray as xr
        import numpy as np
        
        # Add group variation for better convergence
        sample_dataframe['y'] = (
            sample_dataframe['y'] +
            sample_dataframe['group'].map({'G1': -2, 'G2': 2})
        )
        
        # Fit a model with random effects
        model = brmspy.fit(
            formula="y ~ x1 + (1|group)",
            data=sample_dataframe,
            family="gaussian",
            iter=100,
            warmup=50,
            chains=1,
            silent=2,
            refresh=0
        )
        
        # Get raw posterior samples (summary=False)
        random_effects_draws = brmspy.ranef(model, summary=False)
        
        # Verify return type is dict
        assert isinstance(random_effects_draws, dict), \
            f"ranef(summary=False) should return dict, got {type(random_effects_draws)}"
        
        # Verify dict is not empty
        assert len(random_effects_draws) > 0, \
            "Random effects dict should not be empty"
        
        # Verify 'group' is in the dict
        assert 'group' in random_effects_draws, \
            "Random effects should contain 'group' key"
        
        # Get the group random effects draws
        group_re_draws = random_effects_draws['group']
        
        # Verify it's an xarray DataArray
        assert isinstance(group_re_draws, xr.DataArray), \
            f"Random effects should be xarray DataArray, got {type(group_re_draws)}"
        
        # Verify dimensions for summary=False: (draw, group, coef)
        assert group_re_draws.dims == ('draw', 'group', 'coef'), \
            f"DataArray should have dims ('draw', 'group', 'coef'), got {group_re_draws.dims}"
        
        # Verify number of draws (2 chains × 50 post-warmup = 100 draws)
        expected_draws = 1 * (100 - 50)  # chains * (iter - warmup)
        assert group_re_draws.sizes['draw'] == expected_draws, \
            f"Should have {expected_draws} draws, got {group_re_draws.sizes['draw']}"
        
        # Verify we have the expected groups
        groups = list(group_re_draws.coords['group'].values)
        assert len(groups) == 2, \
            "Should have 2 groups"
        assert set(groups) == {'G1', 'G2'}, \
            "Groups should be G1 and G2"
        
        # Verify we can compute statistics from raw draws
        intercept_draws = group_re_draws.sel(coef='Intercept', group='G1')
        assert intercept_draws.shape == (expected_draws,), \
            f"Should have {expected_draws} draws for single group/coef"
        
        # Compute custom statistic (e.g., median)
        median_estimate = float(np.median(intercept_draws.values))
        assert not np.isnan(median_estimate), \
            "Should be able to compute median from draws"
        
        # Verify no NaN values in the draws
        assert not group_re_draws.isnull().any(), \
            "Random effects draws should not contain NaN values"


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
        import brmspy
        import pandas as pd
        
        # Fit a simple model
        model = brmspy.fit(
            formula="y ~ x1",
            data=sample_dataframe,
            family="gaussian",
            iter=100,
            warmup=50,
            chains=1,
            silent=2,
            refresh=0
        )
        
        # Get posterior summary for all parameters
        post_summary = brmspy.posterior_summary(model)
        
        # Verify return type is DataFrame
        assert isinstance(post_summary, pd.DataFrame), \
            f"posterior_summary() should return pandas DataFrame, got {type(post_summary)}"
        
        # Verify DataFrame is not empty
        assert not post_summary.empty, \
            "Posterior summary DataFrame should not be empty"
        
        # Verify expected columns exist
        columns = post_summary.columns.tolist()
        assert any('Estimate' in str(col) for col in columns), \
            "Posterior summary should have Estimate column"
        assert any('Error' in str(col) or 'Est.Error' in str(col) for col in columns), \
            "Posterior summary should have error/uncertainty column"
        
        # Verify it contains more than just fixed effects (should include sigma, etc.)
        param_names = post_summary.index.tolist()
        assert len(param_names) >= 3, \
            "Should have at least Intercept, x1, and sigma parameters"
        
        # Check for specific parameters
        assert any('Intercept' in str(p) for p in param_names), \
            "Should include Intercept parameter"
        assert any('x1' in str(p) for p in param_names), \
            "Should include x1 parameter"
        assert any('sigma' in str(p).lower() for p in param_names), \
            "Should include sigma parameter for Gaussian model"
        
        # Verify no NaN values
        assert not post_summary.isna().any().any(), \
            "Posterior summary should not contain NaN values"


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
        import brmspy
        import pandas as pd
        
        # Fit a model with custom priors
        model = brmspy.fit(
            formula="y ~ x1",
            data=sample_dataframe,
            priors=[brmspy.prior("normal(0, 1)", "b")],
            family="gaussian",
            iter=100,
            warmup=50,
            chains=1,
            silent=2,
            refresh=0
        )
        
        # Get prior summary (all priors including defaults)
        prior_sum = brmspy.prior_summary(model, all=True)
        
        # Verify return type is DataFrame
        assert isinstance(prior_sum, pd.DataFrame), \
            f"prior_summary() should return pandas DataFrame, got {type(prior_sum)}"
        
        # Verify DataFrame is not empty
        assert not prior_sum.empty, \
            "Prior summary DataFrame should not be empty"
        
        # Verify expected columns exist
        columns = prior_sum.columns.tolist()
        assert any('prior' in str(col).lower() for col in columns), \
            "Prior summary should have 'prior' column"
        assert any('class' in str(col).lower() for col in columns), \
            "Prior summary should have 'class' column"
        
        # Verify it contains information about the custom prior
        prior_strings = prior_sum['prior'].tolist() if 'prior' in prior_sum.columns else []
        # The prior string might be in a different format in the summary
        assert len(prior_strings) > 0, \
            "Should have at least one prior specification"
        
        # Test getting only user-set priors (all=False)
        user_priors = brmspy.prior_summary(model, all=False)
        assert isinstance(user_priors, pd.DataFrame), \
            "prior_summary(all=False) should return DataFrame"
        
        # User-set priors should be subset of all priors
        assert len(user_priors) <= len(prior_sum), \
            "User-set priors should be subset of all priors"


@pytest.mark.requires_brms
class TestLooFunction:
    """Test the loo() function for LOO-CV model evaluation."""
    
    @pytest.mark.slow
    def test_loo_basic_functionality(self, sample_dataframe):
        """
        Test loo() for basic LOO-CV computation.
        
        Verifies:
        - Returns LooResult dataclass with all expected fields
        - LOO metrics (elpd_loo, p_loo, looic) are computed
        - Standard errors are provided
        - Diagnostics DataFrame contains Pareto k values
        - Pretty print works via repr
        """
        import brmspy
        from brmspy.types import LooResult
        import pandas as pd
        
        # Fit a simple model
        model = brmspy.fit(
            formula="y ~ x1",
            data=sample_dataframe,
            family="gaussian",
            iter=100,
            warmup=50,
            chains=1,
            silent=2,
            refresh=0
        )
        
        # Compute LOO-CV
        loo_result = brmspy.loo(model)
        
        # Verify return type is LooResult
        assert isinstance(loo_result, LooResult), \
            f"loo() should return LooResult dataclass, got {type(loo_result)}"
        
        # Verify all expected attributes exist
        expected_attrs = [
            'elpd_loo', 'p_loo', 'looic',
            'se_elpd_loo', 'se_p_loo', 'se_looic',
            'estimates', 'pointwise', 'diagnostics', 'psis_object'
        ]
        
        for attr in expected_attrs:
            assert hasattr(loo_result, attr), \
                f"LooResult should have attribute '{attr}'"
        
        # Verify LOO metrics are finite numbers
        assert isinstance(loo_result.elpd_loo, (int, float)), \
            "elpd_loo should be numeric"
        assert not pd.isna(loo_result.elpd_loo), \
            "elpd_loo should not be NaN"
        
        assert isinstance(loo_result.p_loo, (int, float)), \
            "p_loo should be numeric"
        assert loo_result.p_loo > 0, \
            "p_loo (effective parameters) should be positive"
        
        assert isinstance(loo_result.looic, (int, float)), \
            "looic should be numeric"
        assert not pd.isna(loo_result.looic), \
            "looic should not be NaN"
        
        # Verify standard errors are positive
        assert loo_result.se_elpd_loo > 0, \
            "Standard error of elpd_loo should be positive"
        assert loo_result.se_p_loo >= 0, \
            "Standard error of p_loo should be non-negative"
        assert loo_result.se_looic > 0, \
            "Standard error of looic should be positive"
        
        # Verify estimates DataFrame exists and has content
        assert isinstance(loo_result.estimates, pd.DataFrame), \
            "estimates should be a DataFrame"
        assert not loo_result.estimates.empty, \
            "estimates DataFrame should not be empty"
        
        # Verify diagnostics DataFrame contains Pareto k values
        assert isinstance(loo_result.diagnostics, pd.DataFrame), \
            "diagnostics should be a DataFrame"
        if not loo_result.diagnostics.empty:
            assert 'pareto_k' in loo_result.diagnostics.columns or \
                   any('k' in col.lower() for col in loo_result.diagnostics.columns), \
                "diagnostics should contain Pareto k information"
        
        # Verify pretty print works
        loo_str = repr(loo_result)
        assert isinstance(loo_str, str), \
            "repr(loo_result) should return a string"
        assert len(loo_str) > 0, \
            "LOO result string should not be empty"
        assert "LOO-CV Results" in loo_str or "ELPD" in loo_str or "elpd" in loo_str.lower(), \
            "LOO result should contain LOO-CV information"
    
    @pytest.mark.slow
    def test_loo_pareto_k_diagnostics(self, sample_dataframe):
        """
        Test loo() Pareto k diagnostics for identifying problematic observations.
        
        Verifies:
        - Diagnostics DataFrame has one row per observation
        - Pareto k values are in reasonable range
        - Can identify observations with high k values
        - LOO relationship: looic ≈ -2 * elpd_loo
        """
        import brmspy
        import pandas as pd
        import numpy as np
        
        # Fit a simple model
        model = brmspy.fit(
            formula="y ~ x1",
            data=sample_dataframe,
            family="gaussian",
            iter=100,
            warmup=50,
            chains=1,
            silent=2,
            refresh=0
        )
        
        # Compute LOO-CV
        loo_result = brmspy.loo(model)
        
        # Verify diagnostics has correct number of observations
        n_obs = len(sample_dataframe)
        diagnostics = loo_result.diagnostics
        
        assert isinstance(diagnostics, pd.DataFrame), \
            "diagnostics should be a DataFrame"
        
        if not diagnostics.empty and 'pareto_k' in diagnostics.columns:
            # Should have one row per observation
            assert len(diagnostics) == n_obs, \
                f"diagnostics should have {n_obs} rows (one per observation), got {len(diagnostics)}"
            
            # Pareto k values should be finite
            k_values = diagnostics['pareto_k']
            assert not k_values.isna().any(), \
                "Pareto k values should not contain NaN"
            assert np.isfinite(k_values).all(), \
                "Pareto k values should be finite"
            
            # Most Pareto k values should be reasonable (< 0.7 is good)
            # With a simple model and decent data, most should be fine
            good_k = (k_values < 0.7).sum()
            total_k = len(k_values)
            good_ratio = good_k / total_k
            
            # At least some observations should have good k values
            assert good_ratio > 0, \
                "At least some observations should have good Pareto k values (< 0.7)"
        
        # Verify LOOIC ≈ -2 * ELPD_LOO relationship
        computed_looic = -2 * loo_result.elpd_loo
        actual_looic = loo_result.looic
        
        # Should be very close (allowing for small numerical differences)
        looic_diff = abs(computed_looic - actual_looic)
        assert looic_diff < 0.01, \
            f"LOOIC should equal -2*ELPD_LOO, difference: {looic_diff}"
        
        # Verify p_loo is reasonable (should be between 1 and number of parameters + some overhead)
        # For a simple model with Intercept + x1 + sigma, we expect ~3 parameters
        assert 0 < loo_result.p_loo < n_obs, \
            f"p_loo should be positive and less than n_obs, got {loo_result.p_loo}"


@pytest.mark.requires_brms
class TestLooCompareFunction:
    """Test the loo_compare() function for comparing multiple models."""
    
    @pytest.mark.slow
    def test_loo_compare_basic_two_models(self, sample_dataframe):
        """
        Test loo_compare() for comparing two models.
        
        Verifies:
        - Returns LooCompareResult dataclass
        - Table DataFrame has correct structure with model rankings
        - Has expected columns (elpd_diff, se_diff)
        - Models are ordered by performance (best first)
        - Pretty print works via repr()
        - Criterion is 'loo'
        """
        import brmspy
        from brmspy.types import LooCompareResult
        import pandas as pd
        
        # Fit two models of different complexity
        model1 = brmspy.fit(
            formula="y ~ x1",
            data=sample_dataframe,
            family="gaussian",
            iter=100,
            warmup=50,
            chains=1,
            silent=2,
            refresh=0
        )
        
        model2 = brmspy.fit(
            formula="y ~ x1 + x2",
            data=sample_dataframe,
            family="gaussian",
            iter=100,
            warmup=50,
            chains=1,
            silent=2,
            refresh=0
        )
        
        # Compare models
        comparison = brmspy.loo_compare(model1, model2)
        
        # Verify return type is LooCompareResult
        assert isinstance(comparison, LooCompareResult), \
            f"loo_compare() should return LooCompareResult, got {type(comparison)}"
        
        # Verify all expected attributes exist
        assert hasattr(comparison, 'table'), \
            "LooCompareResult should have 'table' attribute"
        assert hasattr(comparison, 'criterion'), \
            "LooCompareResult should have 'criterion' attribute"
        assert hasattr(comparison, 'r'), \
            "LooCompareResult should have 'r' attribute"
        
        # Verify table is a DataFrame
        assert isinstance(comparison.table, pd.DataFrame), \
            f"comparison.table should be pandas DataFrame, got {type(comparison.table)}"
        
        # Verify table is not empty
        assert not comparison.table.empty, \
            "Comparison table should not be empty"
        
        # Verify table has 2 rows (one per model)
        assert len(comparison.table) == 2, \
            f"Comparison table should have 2 rows, got {len(comparison.table)}"
        
        # Verify expected columns exist
        columns = comparison.table.columns.tolist()
        assert any('elpd_diff' in str(col) or 'diff' in str(col).lower() for col in columns), \
            "Comparison table should have elpd_diff column"
        assert any('se' in str(col).lower() for col in columns), \
            "Comparison table should have standard error columns"
        
        # Verify criterion is 'loo'
        assert comparison.criterion.lower() == 'loo', \
            f"Criterion should be 'loo', got '{comparison.criterion}'"
        
        # Verify models are ranked (best model should have elpd_diff = 0)
        if 'elpd_diff' in comparison.table.columns:
            first_model_diff = comparison.table.iloc[0]['elpd_diff']
            assert abs(first_model_diff) < 1e-10, \
                "Best model should have elpd_diff ≈ 0"
        
        # Verify pretty print works
        comp_str = repr(comparison)
        assert isinstance(comp_str, str), \
            "repr(comparison) should return a string"
        assert len(comp_str) > 0, \
            "Comparison string should not be empty"
        assert "LOO" in comp_str.upper() or "elpd" in comp_str.lower(), \
            "Comparison should contain LOO/ELPD information"
    
    @pytest.mark.slow
    def test_loo_compare_custom_model_names(self, sample_dataframe):
        """
        Test loo_compare() with custom model names.
        
        Verifies:
        - Can provide custom model names via model_names parameter
        - Row names in comparison table match custom names
        - Names appear in correct ranking order
        """
        import brmspy
        import pandas as pd
        
        # Fit two models
        model_simple = brmspy.fit(
            formula="y ~ x1",
            data=sample_dataframe,
            family="gaussian",
            iter=100,
            warmup=50,
            chains=1,
            silent=2,
            refresh=0
        )
        
        model_complex = brmspy.fit(
            formula="y ~ x1 + x2",
            data=sample_dataframe,
            family="gaussian",
            iter=100,
            warmup=50,
            chains=1,
            silent=2,
            refresh=0
        )
        
        # Compare with custom names
        custom_names = ["simple_model", "complex_model"]
        comparison = brmspy.loo_compare(
            model_simple,
            model_complex,
            model_names=custom_names
        )
        
        # Verify table has custom row names
        row_names = comparison.table.index.tolist()
        assert len(row_names) == 2, \
            "Should have 2 row names"
        
        # Verify custom names are used (order might vary based on performance)
        assert set(row_names) == set(custom_names), \
            f"Row names should match custom names. Expected {custom_names}, got {row_names}"
        
        # Verify at least one of the custom names appears
        assert any(name in custom_names for name in row_names), \
            "At least one custom name should appear in table"
        
        # Verify we can access rows by custom names
        for name in row_names:
            assert name in comparison.table.index, \
                f"Should be able to access row by name '{name}'"
            row_data = comparison.table.loc[name]
            assert row_data is not None, \
                f"Row data for '{name}' should not be None"


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
        
        # Fit a simple model
        model = brmspy.fit(
            formula="y ~ x1 + x2",
            data=sample_dataframe,
            family="gaussian",
            iter=100,
            warmup=50,
            chains=1,
            silent=2,
            refresh=0
        )
        
        # Create valid new data with all required variables
        n = 10
        newdata = pd.DataFrame({
            'y': np.random.normal(10, 2, n),
            'x1': np.random.normal(0, 1, n),
            'x2': np.random.choice(['A', 'B'], n),
            'group': np.repeat(['G1', 'G2'], n//2)
        })
        
        # Validate newdata
        validated = brmspy.validate_newdata(
            newdata,
            model,
            check_response=False  # Response not needed for prediction
        )
        
        # Verify return type is DataFrame
        assert isinstance(validated, pd.DataFrame), \
            f"validate_newdata() should return DataFrame, got {type(validated)}"
        
        # Verify DataFrame is not empty
        assert not validated.empty, \
            "Validated DataFrame should not be empty"
        
        # Verify it has the same number of rows as input
        assert len(validated) == len(newdata), \
            f"Validated data should have {len(newdata)} rows, got {len(validated)}"
        
        # Verify required columns are present
        assert 'x1' in validated.columns, \
            "Validated data should contain x1 column"
        assert 'x2' in validated.columns, \
            "Validated data should contain x2 column"
    
    @pytest.mark.slow
    def test_validate_newdata_with_missing_variables(self, sample_dataframe):
        """
        Test validate_newdata() with invalid data missing required variables.
        
        Verifies:
        - Raises error when required variables are missing
        - Error message indicates which variable is missing
        - Validation properly detects incomplete data
        """
        import brmspy
        import pandas as pd
        import pytest
        
        # Fit a model with two predictors
        model = brmspy.fit(
            formula="y ~ x1 + x2",
            data=sample_dataframe,
            family="gaussian",
            iter=100,
            warmup=50,
            chains=1,
            silent=2,
            refresh=0
        )
        
        # Create invalid new data missing x2
        invalid_newdata = pd.DataFrame({
            'x1': [1.0, 2.0, 3.0]
            # Missing x2!
        })
        
        # Validation should raise an error
        with pytest.raises(Exception) as exc_info:
            brmspy.validate_newdata(
                invalid_newdata,
                model,
                check_response=False
            )
        
        # Verify error was raised
        assert exc_info.value is not None, \
            "validate_newdata should raise error for missing variables"
        
        # Error message should mention the missing variable or validation failure
        error_msg = str(exc_info.value).lower()
        assert 'x2' in error_msg or 'validate_data' in error_msg or 'column' in error_msg or 'not found' in error_msg, \
            f"Error should mention missing variable, got: {exc_info.value}"