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
            iter=200,
            warmup=100,
            chains=2,
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
            iter=200,
            warmup=100,
            chains=2,
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
            iter=200,
            warmup=100,
            chains=2,
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