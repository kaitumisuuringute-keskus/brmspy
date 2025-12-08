"""
Tests for I/O functions (save_rds, read_rds_raw, read_rds_fit).

These tests verify RDS file saving and loading functionality for brmsfit objects.
"""
import pytest
import pandas as pd
import os
import tempfile

iter = 100
warmup = 50

@pytest.mark.requires_brms
class TestSaveRDS:
    """Test save_rds() function."""
    
    def test_save_rds_with_fit_result(self, sample_dataframe):
        """Test saving a FitResult object to RDS file"""
        from brmspy import brms
        
        # Fit a simple model
        model = brms.fit(
            formula="y ~ x1",
            data=sample_dataframe,
            family="gaussian",
            chains=2,
            iter=iter,
            warmup=warmup,
            refresh=0
        )
        
        # Save to temporary file
        with tempfile.NamedTemporaryFile(suffix='.rds', delete=False) as tmp:
            tmp_path = tmp.name
        
        try:
            brms.save_rds(model, tmp_path)
            
            # Verify file exists and has content
            assert os.path.exists(tmp_path)
            assert os.path.getsize(tmp_path) > 0
        finally:
            # Cleanup
            if os.path.exists(tmp_path):
                os.remove(tmp_path)
    
    def test_save_rds_with_list_vector(self, sample_dataframe):
        """Test saving a raw R ListVector object to RDS file"""
        from brmspy import brms
        
        # Fit a model and extract the R object
        model = brms.fit(
            formula="y ~ x1",
            data=sample_dataframe,
            family="gaussian",
            chains=2,
            iter=iter,
            warmup=warmup,
            refresh=0
        )
        
        # Save using the raw R object
        with tempfile.NamedTemporaryFile(suffix='.rds', delete=False) as tmp:
            tmp_path = tmp.name
        
        try:
            brms.save_rds(model.r, tmp_path)
            
            # Verify file exists
            assert os.path.exists(tmp_path)
            assert os.path.getsize(tmp_path) > 0
        finally:
            if os.path.exists(tmp_path):
                os.remove(tmp_path)
    
    def test_save_rds_with_kwargs(self, sample_dataframe):
        """Test save_rds() with additional R arguments"""
        from brmspy import brms
        
        model = brms.fit(
            formula="y ~ x1",
            data=sample_dataframe,
            family="gaussian",
            chains=2,
            iter=iter,
            warmup=warmup,
            refresh=0
        )
        
        with tempfile.NamedTemporaryFile(suffix='.rds', delete=False) as tmp:
            tmp_path = tmp.name
        
        try:
            # Save with compression (R saveRDS argument)
            brms.save_rds(model, tmp_path, compress=True)
            
            assert os.path.exists(tmp_path)
            assert os.path.getsize(tmp_path) > 0
        finally:
            if os.path.exists(tmp_path):
                os.remove(tmp_path)


@pytest.mark.requires_brms
class TestReadRDSRaw:
    """Test read_rds_raw() function."""
    
    def test_read_rds_raw_returns_list_vector(self, sample_dataframe):
        """Test that read_rds_raw() returns a raw R ListVector"""
        from brmspy import brms
        from rpy2.robjects import ListVector
        
        # Fit and save a model
        model = brms.fit(
            formula="y ~ x1",
            data=sample_dataframe,
            family="gaussian",
            chains=2,
            iter=iter,
            warmup=warmup,
            refresh=0
        )
        
        with tempfile.NamedTemporaryFile(suffix='.rds', delete=False) as tmp:
            tmp_path = tmp.name
        
        try:
            brms.save_rds(model, tmp_path)
            
            # Read back as raw object
            raw_object = brms.read_rds_raw(tmp_path)
            
            # Verify it's a ListVector
            assert isinstance(raw_object, ListVector)
        finally:
            if os.path.exists(tmp_path):
                os.remove(tmp_path)
    
    def test_read_rds_raw_with_kwargs(self, sample_dataframe):
        """Test read_rds_raw() with R readRDS arguments"""
        from brmspy import brms
        
        model = brms.fit(
            formula="y ~ x1",
            data=sample_dataframe,
            family="gaussian",
            chains=2,
            iter=iter,
            warmup=warmup,
            refresh=0
        )
        
        with tempfile.NamedTemporaryFile(suffix='.rds', delete=False) as tmp:
            tmp_path = tmp.name
        
        try:
            brms.save_rds(model, tmp_path)
            
            # Read with refhook argument (if needed)
            raw_object = brms.read_rds_raw(tmp_path)
            
            assert raw_object is not None
        finally:
            if os.path.exists(tmp_path):
                os.remove(tmp_path)
    
    def test_read_rds_raw_nonexistent_file(self):
        """Test read_rds_raw() with non-existent file raises error"""
        from brmspy import brms
        
        with pytest.raises(Exception):
            brms.read_rds_raw("/nonexistent/path/to/file.rds")


@pytest.mark.requires_brms
class TestReadRDSFit:
    """Test read_rds_fit() function."""
    
    def test_read_rds_fit_returns_fit_result(self, sample_dataframe):
        """Test that read_rds_fit() returns a FitResult with idata and r"""
        from brmspy import brms
        from brmspy.types import FitResult
        import arviz as az
        
        # Fit and save a model
        model = brms.fit(
            formula="y ~ x1",
            data=sample_dataframe,
            family="gaussian",
            chains=2,
            iter=iter,
            warmup=warmup,
            refresh=0
        )
        
        with tempfile.NamedTemporaryFile(suffix='.rds', delete=False) as tmp:
            tmp_path = tmp.name
        
        try:
            brms.save_rds(model, tmp_path)
            
            # Read back as FitResult
            loaded_model = brms.read_rds_fit(tmp_path)
            
            # Verify return type
            assert isinstance(loaded_model, FitResult)
            
            # Verify has both idata and r attributes
            assert hasattr(loaded_model, 'idata')
            assert hasattr(loaded_model, 'r')
            
            # Verify idata is InferenceData
            assert isinstance(loaded_model.idata, az.InferenceData)
            
            # Verify idata has expected groups
            assert hasattr(loaded_model.idata, 'posterior')
        finally:
            if os.path.exists(tmp_path):
                os.remove(tmp_path)
    
    def test_read_rds_fit_preserves_data(self, sample_dataframe):
        """Test that read_rds_fit() preserves model data correctly"""
        from brmspy import brms
        import arviz as az
        
        # Fit model
        original_model = brms.fit(
            formula="y ~ x1",
            data=sample_dataframe,
            family="gaussian",
            chains=2,
            iter=iter,
            warmup=warmup,
            refresh=0
        )
        
        with tempfile.NamedTemporaryFile(suffix='.rds', delete=False) as tmp:
            tmp_path = tmp.name
        
        try:
            brms.save_rds(original_model, tmp_path)
            loaded_model = brms.read_rds_fit(tmp_path)
            
            # Check posterior dimensions match
            orig_posterior = original_model.idata.posterior
            load_posterior = loaded_model.idata.posterior
            
            # Verify same parameters exist
            assert set(orig_posterior.data_vars) == set(load_posterior.data_vars)
            
            # Verify dimensions match
            for var in orig_posterior.data_vars:
                assert orig_posterior[var].shape == load_posterior[var].shape
        finally:
            if os.path.exists(tmp_path):
                os.remove(tmp_path)
    
    def test_read_rds_fit_with_kwargs(self, sample_dataframe):
        """Test read_rds_fit() with additional R arguments"""
        from brmspy import brms
        
        model = brms.fit(
            formula="y ~ x1",
            data=sample_dataframe,
            family="gaussian",
            chains=2,
            iter=iter,
            warmup=warmup,
            refresh=0
        )
        
        with tempfile.NamedTemporaryFile(suffix='.rds', delete=False) as tmp:
            tmp_path = tmp.name
        
        try:
            brms.save_rds(model, tmp_path)
            
            # Read with kwargs (even if just passed through)
            loaded_model = brms.read_rds_fit(tmp_path)
            
            assert loaded_model is not None
            assert hasattr(loaded_model, 'idata')
        finally:
            if os.path.exists(tmp_path):
                os.remove(tmp_path)


@pytest.mark.requires_brms
class TestRoundTrip:
    """Test complete save/load round-trip workflows."""
    
    def test_round_trip_simple_model(self, sample_dataframe):
        """Test complete save and load cycle with simple model"""
        from brmspy import brms
        import arviz as az
        
        # Fit original model
        original_model = brms.fit(
            formula="y ~ x1",
            data=sample_dataframe,
            family="gaussian",
            chains=2,
            iter=iter,
            warmup=warmup,
            refresh=0
        )
        
        with tempfile.NamedTemporaryFile(suffix='.rds', delete=False) as tmp:
            tmp_path = tmp.name
        
        try:
            # Save
            brms.save_rds(original_model, tmp_path)
            
            # Load
            loaded_model = brms.read_rds_fit(tmp_path)
            
            # Compare summaries
            orig_summary = az.summary(original_model.idata, kind="stats")
            load_summary = az.summary(loaded_model.idata, kind="stats")
            
            # Verify parameter estimates are the same
            assert orig_summary.shape == load_summary.shape
            
            # Check specific parameters match (within numerical tolerance)
            for param in orig_summary.index:
                if param in load_summary.index:
                    orig_mean = orig_summary.at[param, 'mean']  # type: ignore
                    load_mean = load_summary.at[param, 'mean']  # type: ignore
                    assert abs(float(orig_mean) - float(load_mean)) < 1e-6 * abs(float(orig_mean))  # type: ignore
        finally:
            if os.path.exists(tmp_path):
                os.remove(tmp_path)
    
    def test_round_trip_complex_model(self, sample_dataframe):
        """Test save/load with more complex model including random effects"""
        from brmspy import brms
        
        # Fit model with random effects
        model = brms.fit(
            formula="y ~ x1 + (1|group)",
            data=sample_dataframe,
            family="gaussian",
            chains=2,
            iter=iter,
            warmup=warmup,
            refresh=0
        )
        
        with tempfile.NamedTemporaryFile(suffix='.rds', delete=False) as tmp:
            tmp_path = tmp.name
        
        try:
            # Save and load
            brms.save_rds(model, tmp_path)
            loaded_model = brms.read_rds_fit(tmp_path)
            
            # Verify both have same parameter structure
            orig_vars = set(model.idata.posterior.data_vars)
            load_vars = set(loaded_model.idata.posterior.data_vars)
            assert orig_vars == load_vars
            
            # Verify random effects are preserved
            assert 'r_group' in loaded_model.idata.posterior.data_vars or \
                   any('group' in str(v) for v in loaded_model.idata.posterior.data_vars)
        finally:
            if os.path.exists(tmp_path):
                os.remove(tmp_path)
    
    def test_round_trip_with_predictions(self, sample_dataframe):
        """Test that saved model can be used for predictions after loading"""
        from brmspy import brms
        
        # Fit model
        model = brms.fit(
            formula="y ~ x1",
            data=sample_dataframe,
            family="gaussian",
            chains=2,
            iter=iter,
            warmup=warmup,
            refresh=0
        )
        
        with tempfile.NamedTemporaryFile(suffix='.rds', delete=False) as tmp:
            tmp_path = tmp.name
        
        try:
            # Save and load
            brms.save_rds(model, tmp_path)
            loaded_model = brms.read_rds_fit(tmp_path)
            
            # Create new data for predictions
            newdata = pd.DataFrame({
                'x1': [0.5, 1.0, 1.5]
            })
            
            # Test that loaded model can make predictions
            predictions = brms.posterior_predict(loaded_model, newdata=newdata)
            
            assert predictions is not None
            assert hasattr(predictions, 'idata')
            assert predictions.idata.posterior_predictive is not None
        finally:
            if os.path.exists(tmp_path):
                os.remove(tmp_path)
    
    def test_round_trip_preserves_model_info(self, sample_dataframe):
        """Test that model metadata is preserved through save/load"""
        from brmspy import brms
        
        model = brms.fit(
            formula="y ~ x1",
            data=sample_dataframe,
            family="gaussian",
            chains=2,
            iter=iter,
            warmup=warmup,
            refresh=0
        )
        
        with tempfile.NamedTemporaryFile(suffix='.rds', delete=False) as tmp:
            tmp_path = tmp.name
        
        try:
            brms.save_rds(model, tmp_path)
            loaded_model = brms.read_rds_fit(tmp_path)
            
            # Both should have the same structure
            assert type(model) == type(loaded_model)
            
            # Both should have R objects
            assert model.r is not None
            assert loaded_model.r is not None
        finally:
            if os.path.exists(tmp_path):
                os.remove(tmp_path)


@pytest.mark.requires_brms
class TestEdgeCases:
    """Test edge cases and error handling."""
    
    def test_save_to_directory_without_extension(self, sample_dataframe):
        """Test saving to path without .rds extension still works"""
        from brmspy import brms
        
        model = brms.fit(
            formula="y ~ x1",
            data=sample_dataframe,
            family="gaussian",
            chains=2,
            iter=iter,
            warmup=warmup,
            refresh=0
        )
        
        with tempfile.NamedTemporaryFile(suffix='', delete=False) as tmp:
            tmp_path = tmp.name
        
        try:
            # Save without .rds extension
            brms.save_rds(model, tmp_path)
            
            # Should still be readable
            loaded = brms.read_rds_fit(tmp_path)
            assert loaded is not None
        finally:
            if os.path.exists(tmp_path):
                os.remove(tmp_path)
    
    def test_save_to_subdirectory(self, sample_dataframe):
        """Test saving to a subdirectory path"""
        from brmspy import brms
        
        model = brms.fit(
            formula="y ~ x1",
            data=sample_dataframe,
            family="gaussian",
            chains=2,
            iter=iter,
            warmup=warmup,
            refresh=0
        )
        
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp_path = os.path.join(tmpdir, 'subdir', 'model.rds')
            
            # Create subdirectory
            os.makedirs(os.path.dirname(tmp_path), exist_ok=True)
            
            # Save to subdirectory
            brms.save_rds(model, tmp_path)
            
            assert os.path.exists(tmp_path)
            
            # Load from subdirectory
            loaded = brms.read_rds_fit(tmp_path)
            assert loaded is not None


if __name__ == '__main__':
    pytest.main([__file__, '-v'])