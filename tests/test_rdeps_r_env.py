"""
Tests for brmspy.runtime R environment operations.

Focus: R environment state management, GitHub token forwarding, package operations.

These tests cover R-specific environment operations that are unique to this module.
"""

from pathlib import Path
from tempfile import TemporaryDirectory
import pytest
import os


@pytest.mark.rdeps
class TestGitHubTokenForwarding:
    """Test GitHub token forwarding to R environment."""
    
    def test_forward_token_when_set(self):
        """Forward GITHUB_TOKEN when set"""
        from brmspy.runtime._r_env import forward_github_token
        import rpy2.robjects as ro
        from typing import cast
        
        original_token = os.environ.get('GITHUB_TOKEN')
        original_pat = os.environ.get('GITHUB_PAT')
        try:
            # Clear PAT, set TOKEN
            if 'GITHUB_PAT' in os.environ:
                del os.environ['GITHUB_PAT']
            os.environ['GITHUB_TOKEN'] = 'test_token_12345'
            
            forward_github_token()
            
            # Verify token was set in R
            r_token = str(cast(list, ro.r('Sys.getenv("GITHUB_TOKEN")'))[0])
            assert r_token == 'test_token_12345'
            
        finally:
            if original_token:
                os.environ['GITHUB_TOKEN'] = original_token
            elif 'GITHUB_TOKEN' in os.environ:
                del os.environ['GITHUB_TOKEN']
            
            if original_pat:
                os.environ['GITHUB_PAT'] = original_pat
            elif 'GITHUB_PAT' in os.environ:
                del os.environ['GITHUB_PAT']
    
    def test_forward_pat_preferred_over_token(self):
        """Forward GITHUB_PAT when both PAT and TOKEN are set"""
        from brmspy.runtime._r_env import forward_github_token
        import rpy2.robjects as ro
        from typing import cast
        
        original_token = os.environ.get('GITHUB_TOKEN')
        original_pat = os.environ.get('GITHUB_PAT')
        try:
            os.environ['GITHUB_TOKEN'] = 'test_token'
            os.environ['GITHUB_PAT'] = 'test_pat_preferred'
            
            forward_github_token()
            
            # PAT should be set, not TOKEN
            r_pat = str(cast(list, ro.r('Sys.getenv("GITHUB_PAT")'))[0])
            assert r_pat == 'test_pat_preferred'
            
        finally:
            if original_token:
                os.environ['GITHUB_TOKEN'] = original_token
            elif 'GITHUB_TOKEN' in os.environ:
                del os.environ['GITHUB_TOKEN']
            
            if original_pat:
                os.environ['GITHUB_PAT'] = original_pat
            elif 'GITHUB_PAT' in os.environ:
                del os.environ['GITHUB_PAT']
    
    def test_forward_token_when_not_set(self):
        """Handle case when neither token is set"""
        from brmspy.runtime._r_env import forward_github_token
        
        original_token = os.environ.get('GITHUB_TOKEN')
        original_pat = os.environ.get('GITHUB_PAT')
        try:
            if 'GITHUB_TOKEN' in os.environ:
                del os.environ['GITHUB_TOKEN']
            if 'GITHUB_PAT' in os.environ:
                del os.environ['GITHUB_PAT']
            
            # Should not raise exception
            forward_github_token()
            
        finally:
            if original_token:
                os.environ['GITHUB_TOKEN'] = original_token
            if original_pat:
                os.environ['GITHUB_PAT'] = original_pat


@pytest.mark.rdeps
class TestPackageStateChecks:
    """Test R package state detection."""
    
    def test_namespace_loaded_vs_attached(self):
        """Verify distinction between namespace loaded and package attached"""
        from brmspy.runtime._r_env import is_namespace_loaded, is_package_attached
        import rpy2.robjects as ro
        
        # Load stats with library (loads and attaches)
        ro.r('library(stats)')
        
        # Both should be true for stats
        assert is_namespace_loaded("stats") is True
        assert is_package_attached("stats") is True
        
        # Load tools with requireNamespace (loads but doesn't attach)
        ro.r('requireNamespace("tools", quietly=TRUE)')
        
        # Namespace should be loaded
        assert is_namespace_loaded("tools") is True
        
        # Non-existent package
        assert is_namespace_loaded("nonexistent_pkg_xyz") is False
        assert is_package_attached("nonexistent_pkg_xyz") is False
    
    def test_unload_package_not_loaded(self):
        """Unload package that isn't loaded"""
        from brmspy.runtime._r_env import unload_package
        
        # Try to unload a package that's not loaded
        # Should return False but not raise exception
        result = unload_package("nonexistent_package_xyz")
        assert isinstance(result, bool)
    
    def test_unload_package_loaded(self):
        """Attempt to unload a loaded package"""
        from brmspy.runtime._r_env import unload_package, is_namespace_loaded
        import rpy2.robjects as ro
        
        # Load a package
        ro.r('library(tools)')
        assert is_namespace_loaded("tools") is True
        
        # Try to unload it
        result = unload_package("tools")
        
        # Result should be boolean (success/failure depends on dependencies)
        assert isinstance(result, bool)


@pytest.mark.rdeps
class TestPackageQueries:
    """Test R package query operations."""
    
    def test_get_package_version_installed(self):
        """Get version of installed package"""
        from brmspy.runtime._r_packages import get_package_version
        
        # stats should always be available
        version = get_package_version("stats")
        
        if version is not None:
            assert isinstance(version, str)
            # Should have version format like "4.3.0"
            parts = version.split(".")
            assert len(parts) >= 2
    
    def test_get_package_version_not_installed(self):
        """Return None for non-installed package"""
        from brmspy.runtime._r_packages import get_package_version
        
        version = get_package_version("nonexistent_package_xyz_123")
        assert version is None
    
    def test_is_package_installed_true(self):
        """Return True for installed package"""
        from brmspy.runtime._r_packages import is_package_installed
        
        # stats package should always be available
        result = is_package_installed("stats")
        assert result is True
    
    def test_is_package_installed_false(self):
        """Return False for non-installed package"""
        from brmspy.runtime._r_packages import is_package_installed
        
        result = is_package_installed("nonexistent_package_xyz_123")
        assert result is False


@pytest.mark.rdeps
class TestLibPathOperations:
    """Test R library path operations."""
    
    def test_get_lib_paths_returns_list(self):
        """get_lib_paths returns list of strings"""
        from brmspy.runtime._r_env import get_lib_paths
        
        paths = get_lib_paths()
        
        assert isinstance(paths, list)
        assert len(paths) > 0
        assert all(isinstance(p, str) for p in paths)
    
    def test_set_lib_paths_modifies_r_env(self):
        """set_lib_paths modifies R environment"""
        from brmspy.runtime._r_env import get_lib_paths, set_lib_paths
        
        # Save original
        original_paths = get_lib_paths()
        
        try:
            # Set new paths (add a fake path)
            with TemporaryDirectory() as dir:
                test_path = os.path.realpath(dir)
                test_paths = [test_path] + original_paths
                set_lib_paths(test_paths)
                
                # Verify change
                new_paths = get_lib_paths()
                assert any(
                    os.path.samefile(test_path, p)
                    for p in new_paths
                )
            
        finally:
            # Restore original
            set_lib_paths(original_paths)


@pytest.mark.rdeps
class TestCmdStanPath:
    """Test CmdStan path operations."""
    
    def test_get_cmdstan_path_returns_optional_str(self):
        """get_cmdstan_path returns str or None"""
        from brmspy.runtime._r_env import get_cmdstan_path
        
        path = get_cmdstan_path()
        
        # Should be None or a string path
        assert path is None or isinstance(path, str)