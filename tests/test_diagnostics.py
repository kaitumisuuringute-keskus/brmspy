import pytest

@pytest.mark.requires_brms
class TestAdditionalFunctions:
    @pytest.mark.slow
    def test_summary_function(self, sample_dataframe):
        """
        Test summary() function for model summaries.
        
        Tests that summary():
        - Returns a pandas DataFrame
        - Contains expected columns (Estimate, Est.Error, etc.)
        - Provides summary statistics for model parameters
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
            chains=1,
            silent=2,
            refresh=0
        )
        
        # Get summary
        summary_df = brmspy.summary(model)
        
        # Verify return type
        assert isinstance(summary_df, pd.DataFrame), \
            "summary() should return a pandas DataFrame"
        
        # Verify DataFrame is not empty
        assert len(summary_df) > 0, \
            "Summary DataFrame should have rows"
        assert len(summary_df.columns) > 0, \
            "Summary DataFrame should have columns"
        
        # Check for expected columns (brms typically includes Estimate, Est.Error, etc.)
        # The exact column names may vary by brms version
        assert any('Estimate' in col or 'estimate' in col.lower() for col in summary_df.columns), \
            "Summary should contain estimate information"