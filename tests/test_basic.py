"""
Basic unit tests for brmspy that don't require R/brms installation.

These tests check:
- Module imports
- Version information
- Data structure conversions
- Helper functions
"""

import pytest


class TestImports:
    """Test that all modules and functions can be imported."""

    def test_import_brmspy(self):
        """Test basic brmspy import"""
        import brmspy

        assert brmspy is not None

    def test_version_exists(self):
        """Test that version is defined"""
        import brmspy

        assert hasattr(brmspy, "__version__")
        assert isinstance(brmspy.__version__, str)

    def test_main_functions_exist(self):
        """Test that main API functions are accessible"""
        import brmspy
        import brmspy.brms

        assert not hasattr(brmspy.brms, "install_brms")
        assert hasattr(brmspy.brms, "get_brms_version")
        assert hasattr(brmspy.brms, "get_brms_data")
        assert hasattr(brmspy.brms, "brm")
        assert callable(brmspy.brms.get_brms_version)
        assert callable(brmspy.brms.get_brms_data)
        assert callable(brmspy.brms.brm)

        from brmspy import brms

        assert hasattr(brms, "brm")
        assert callable(brms.brm)


class TestErrorHandling:
    """Test error handling and messages."""

    @pytest.mark.worker
    def test_brms_not_installed_error_message(self):
        """Test that helpful error is raised when brms is not found"""
        import rpy2.robjects.packages as rpackages
        from brmspy._runtime._state import get_brms

        # This test might pass if brms IS installed
        # So we'll just check the function exists and can be called
        try:
            brms = get_brms()
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
        from brmspy import brms

        assert hasattr(brms, "__all__")
        assert isinstance(brms.__all__, list)

        # Check key functions are in __all__
        assert "fit" in brms.__all__
        assert "get_brms_data" in brms.__all__

    def test_submodule_structure(self):
        """Test submodule structure"""
        from brmspy import brms as brmspy_module

        # Check module has expected functions
        assert not hasattr(brmspy_module, "install_brms")
        assert hasattr(brmspy_module, "fit")
        assert hasattr(brmspy_module, "get_brms_data")


class TestDocumentation:
    """Test that functions have proper documentation."""

    def test_fit_has_docstring(self):
        """Test fit() has comprehensive docstring"""
        from brmspy import brms

        assert brms.fit.__doc__ is not None
        assert len(brms.fit.__doc__) > 100
        assert "Parameters" in brms.fit.__doc__
        assert "Returns" in brms.fit.__doc__
        assert "Examples" in brms.fit.__doc__

    def test_get_brms_data_has_docstring(self):
        """Test get_brms_data() has docstring"""
        from brmspy import brms

        assert brms.get_brms_data.__doc__ is not None
        assert "dataset_name" in brms.get_brms_data.__doc__

    def test_summary_has_docstring(self):
        """Test summary() has docstring"""
        from brmspy import brms

        assert brms.summary.__doc__ is not None
        assert "summary" in brms.summary.__doc__.lower()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
