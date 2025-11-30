"""
Basic unit tests for brmspy that don't require R/brms installation.

These tests check:
- Module imports
- Version information
- Data structure conversions
- Helper functions
"""
import pytest
import pandas as pd
import numpy as np


class TestImports:
    """Test that all modules and functions can be imported."""

    def test_import_brmspy(self):
        """Test basic brmspy import"""
        import brmspy
        assert brmspy is not None
    
    def test_version_exists(self):
        """Test that version is defined"""
        import brmspy
        assert hasattr(brmspy, '__version__')
        assert isinstance(brmspy.__version__, str)
    
    def test_main_functions_exist(self):
        """Test that main API functions are accessible"""
        import brmspy
        assert hasattr(brmspy, 'install_brms')
        assert hasattr(brmspy, 'get_brms_version')
        assert hasattr(brmspy, 'get_brms_data')
        assert hasattr(brmspy, 'fit')
        assert callable(brmspy.install_brms)
        assert callable(brmspy.get_brms_version)
        assert callable(brmspy.get_brms_data)
        assert callable(brmspy.fit)


class TestDataConversion:
    """Test data conversion utilities."""
    
    def test_convert_dataframe_to_r(self, sample_dataframe):
        """Test DataFrame to R conversion"""
        from brmspy.helpers.conversion import py_to_r
        from rpy2.robjects import DataFrame as RDataFrame
        
        r_data = py_to_r(sample_dataframe)
        assert isinstance(r_data, RDataFrame)
    
    def test_convert_dict_to_r(self, sample_dict):
        """Test dict to R list conversion"""
        from brmspy.helpers.conversion import py_to_r
        from rpy2.robjects import ListVector
        
        r_data = py_to_r(sample_dict)
        assert isinstance(r_data, ListVector)


class TestTypeCoercion:
    """Test Stan type coercion functionality."""
    
    def test_coerce_types_basic(self):
        """Test basic type coercion from Stan code"""
        from brmspy.helpers.conversion import _coerce_stan_types
        
        # Simple Stan code with int and real types
        stan_code = """
        data {
            int N;
            int<lower=0> K;
            real y[N];
            matrix[N, K] X;
        }
        """
        
        # Sample data that needs coercion
        stan_data = {
            'N': np.array([50]),  # Should become scalar int
            'K': np.array([3]),   # Should become scalar int
            'y': np.array([1.5, 2.3, 3.1]),  # Should stay as is
            'X': np.random.randn(3, 3)  # Should stay as is
        }
        
        result = _coerce_stan_types(stan_code, stan_data)
        
        # Check N and K are scalars and integers
        assert isinstance(result['N'], (int, np.integer))
        assert isinstance(result['K'], (int, np.integer))
        assert result['N'] == 50
        assert result['K'] == 3
    
    def test_coerce_types_preserves_arrays(self):
        """Test that arrays are preserved correctly"""
        from brmspy.helpers.conversion import _coerce_stan_types
        
        stan_code = """
        data {
            int N;
            vector[N] y;
        }
        """
        
        y_data = np.array([1.1, 2.2, 3.3, 4.4, 5.5])
        stan_data = {
            'N': np.array([5]),
            'y': y_data
        }
        
        result = _coerce_stan_types(stan_code, stan_data)
        
        # N should be scalar
        assert isinstance(result['N'], (int, np.integer))
        # y should still be an array
        assert isinstance(result['y'], np.ndarray)
        assert len(result['y']) == 5
    
    def test_coerce_int_array(self):
        """Test coercion of int arrays"""
        from brmspy.helpers.conversion import _coerce_stan_types
        
        stan_code = """
        data {
            int N;
            int Y[N];
        }
        """
        
        # Simulate data that might come from R with float types
        stan_data = {
            'N': np.array([5.0]),
            'Y': np.array([1.0, 2.0, 3.0, 4.0, 5.0])  # Floats that should be ints
        }
        
        result = _coerce_stan_types(stan_code, stan_data)
        
        # N should be scalar int
        assert isinstance(result['N'], (int, np.integer))
        assert result['N'] == 5
        
        # Y should be int array
        assert isinstance(result['Y'], np.ndarray)
        assert result['Y'].dtype in [np.int32, np.int64]
        assert np.array_equal(result['Y'], [1, 2, 3, 4, 5])
    
    def test_coerce_mixed_types(self):
        """Test coercion with mixed int and real types"""
        from brmspy.helpers.conversion import _coerce_stan_types

        stan_code = """
        data {
            int<lower=1> N;
            int K;
            int Y[N];
            real X[N];
            vector[N] Z;
        }
        """
        
        stan_data = {
            'N': np.array([10.0]),
            'K': np.array([3.0]),
            'Y': np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0]),
            'X': np.array([1.1, 2.2, 3.3, 4.4, 5.5, 6.6, 7.7, 8.8, 9.9, 10.1]),
            'Z': np.array([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0])
        }
        
        result = _coerce_stan_types(stan_code, stan_data)
        
        # Scalars should be properly coerced
        assert isinstance(result['N'], (int, np.integer))
        assert isinstance(result['K'], (int, np.integer))
        
        # Int array should be int type
        assert result['Y'].dtype in [np.int32, np.int64]
        
        # Real arrays should stay as float
        assert result['X'].dtype in [np.float32, np.float64]
        assert result['Z'].dtype in [np.float32, np.float64]
    
    def test_coerce_handles_non_numpy(self):
        """Test coercion handles non-numpy types"""
        from brmspy.helpers.conversion import _coerce_stan_types
        
        stan_code = """
        data {
            int N;
            int K;
        }
        """
        
        # Python lists instead of numpy arrays
        stan_data = {
            'N': [5],
            'K': [3]
        }
        
        result = _coerce_stan_types(stan_code, stan_data)
        
        # Should convert and coerce properly
        assert isinstance(result['N'], (int, np.integer))
        assert isinstance(result['K'], (int, np.integer))
        assert result['N'] == 5
        assert result['K'] == 3
    
    def test_coerce_new_stan_array_syntax(self):
        """Test type coercion with new Stan array syntax: array[N] int Y
        
        This tests the specific scenario found with epilepsy count data where:
        - R returns count data as float64
        - Stan declares it as array[N] int Y (new syntax)
        - Must be correctly identified and coerced to int
        
        Regression test for issue where old parser captured 'array' as type
        instead of 'int', causing Stan runtime errors.
        """
        from brmspy.helpers.conversion import _coerce_stan_types
        
        # New Stan array syntax
        stan_code = """
        data {
            int<lower=1> N;
            array[N] int Y;
            int K;
        }
        """
        
        # Simulate data from R (floats that should be ints)
        stan_data = {
            'N': np.array([5.0]),
            'Y': np.array([1.0, 2.0, 3.0, 4.0, 5.0]),  # count data as float64
            'K': np.array([3.0])
        }
        
        result = _coerce_stan_types(stan_code, stan_data)
        
        # Verify scalars are converted
        assert isinstance(result['N'], (int, np.integer))
        assert isinstance(result['K'], (int, np.integer))
        
        # Verify array is converted to int type
        assert isinstance(result['Y'], np.ndarray)
        assert result['Y'].dtype in [np.int32, np.int64], \
            f"Y should be int type but got {result['Y'].dtype}"
        
        # Verify values are preserved
        assert np.array_equal(result['Y'], np.array([1, 2, 3, 4, 5]))


class TestErrorHandling:
    """Test error handling and messages."""
    
    def test_brms_not_installed_error_message(self):
        """Test that helpful error is raised when brms is not found"""
        from brmspy.helpers.singleton import _get_brms
        import rpy2.robjects.packages as rpackages
        
        # This test might pass if brms IS installed
        # So we'll just check the function exists and can be called
        try:
            brms = _get_brms()
            # If successful, brms is installed - that's fine
            assert brms is not None
        except ImportError as e:
            # Check error message is helpful
            error_msg = str(e)
            assert "brms R package not found" in error_msg
            assert "install_brms()" in error_msg


class TestModuleStructure:
    """Test module structure and organization."""
    
    def test_all_exports(self):
        """Test that __all__ is properly defined"""
        import brmspy
        assert hasattr(brmspy, '__all__')
        assert isinstance(brmspy.__all__, list)
        
        # Check key functions are in __all__
        assert 'fit' in brmspy.__all__
        assert 'install_brms' in brmspy.__all__
        assert 'get_brms_data' in brmspy.__all__
        assert 'get_brms_version' in brmspy.__all__
    
    def test_submodule_structure(self):
        """Test submodule structure"""
        from brmspy import brms as brmspy_module
        
        # Check module has expected functions
        assert hasattr(brmspy_module, 'install_brms')
        assert hasattr(brmspy_module, 'fit')
        assert hasattr(brmspy_module, 'get_brms_data')


class TestDocumentation:
    """Test that functions have proper documentation."""
    
    def test_fit_has_docstring(self):
        """Test fit() has comprehensive docstring"""
        import brmspy
        assert brmspy.fit.__doc__ is not None
        assert len(brmspy.fit.__doc__) > 100
        assert "Parameters" in brmspy.fit.__doc__
        assert "Returns" in brmspy.fit.__doc__
        assert "Examples" in brmspy.fit.__doc__
    
    def test_install_brms_has_docstring(self):
        """Test install_brms() has docstring"""
        import brmspy
        assert brmspy.install_brms.__doc__ is not None
        assert "version" in brmspy.install_brms.__doc__
        assert "Examples" in brmspy.install_brms.__doc__
    
    def test_get_brms_data_has_docstring(self):
        """Test get_brms_data() has docstring"""
        import brmspy
        assert brmspy.get_brms_data.__doc__ is not None
        assert "dataset_name" in brmspy.get_brms_data.__doc__
    
    def test_summary_has_docstring(self):
        """Test summary() has docstring"""
        import brmspy
        assert brmspy.summary.__doc__ is not None
        assert "summary" in brmspy.summary.__doc__.lower()


if __name__ == '__main__':
    pytest.main([__file__, '-v'])