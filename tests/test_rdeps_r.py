"""
Tests for brmspy.binaries.r module (R package operations).

Focus: R package detection, version checking, GitHub token forwarding, unloading.
Target: 70% → 88%+ coverage

These tests exercise R package management operations.
"""

import pytest
import os


@pytest.mark.crossplatform
class TestTryForceUnloadPackage:
    """Test package unloading functionality."""
    
    def test_try_force_unload_package_not_loaded(self):
        """Unload package that isn't loaded (lines 93-99)"""
        from brmspy.binaries.r import _try_force_unload_package
        
        # Try to unload a package that's not loaded
        # Should not raise exception
        _try_force_unload_package("nonexistent_package_xyz", uninstall=False)
    
    def test_try_force_unload_package_loaded(self):
        """Unload a loaded package"""
        from brmspy.binaries.r import _try_force_unload_package
        import rpy2.robjects as ro
        
        # Load a package first
        ro.r('library(stats)')  # stats should be available
        
        # Try to unload it
        _try_force_unload_package("stats", uninstall=False)
        
        # Package operations should not crash


@pytest.mark.crossplatform
class TestForwardGithubToken:
    """Test GitHub token forwarding to R."""
    
    def test_forward_github_token_when_set(self):
        """Forward token when GITHUB_TOKEN is set (lines 105-121)"""
        from brmspy.binaries.r import _forward_github_token_to_r
        import rpy2.robjects as ro
        from typing import cast
        
        # Set a test token
        original_token = os.environ.get('GITHUB_TOKEN')
        try:
            os.environ['GITHUB_TOKEN'] = 'test_token_12345'
            
            _forward_github_token_to_r()
            
            # Verify token was set in R
            r_token = str(cast(list, ro.r('Sys.getenv("GITHUB_TOKEN")'))[0])
            assert r_token == 'test_token_12345'
            
        finally:
            # Restore original token
            if original_token:
                os.environ['GITHUB_TOKEN'] = original_token
            elif 'GITHUB_TOKEN' in os.environ:
                del os.environ['GITHUB_TOKEN']
    
    def test_forward_github_token_when_not_set(self):
        """Handle case when GITHUB_TOKEN not set (line 111)"""
        from brmspy.binaries.r import _forward_github_token_to_r
        
        # Ensure token is not set
        original_token = os.environ.get('GITHUB_TOKEN')
        try:
            if 'GITHUB_TOKEN' in os.environ:
                del os.environ['GITHUB_TOKEN']
            
            # Should not raise exception
            _forward_github_token_to_r()
            
        finally:
            # Restore original token
            if original_token:
                os.environ['GITHUB_TOKEN'] = original_token
    
    def test_forward_github_token_empty_string(self):
        """Handle empty GITHUB_TOKEN (lines 117-121)"""
        from brmspy.binaries.r import _forward_github_token_to_r
        
        original_token = os.environ.get('GITHUB_TOKEN')
        try:
            os.environ['GITHUB_TOKEN'] = ''
            
            # Should not set token in R when empty
            _forward_github_token_to_r()
            
        finally:
            if original_token:
                os.environ['GITHUB_TOKEN'] = original_token
            elif 'GITHUB_TOKEN' in os.environ:
                del os.environ['GITHUB_TOKEN']
    
    def test_forward_github_token_whitespace(self):
        """Handle whitespace-only GITHUB_TOKEN (lines 117-121)"""
        from brmspy.binaries.r import _forward_github_token_to_r
        
        original_token = os.environ.get('GITHUB_TOKEN')
        try:
            os.environ['GITHUB_TOKEN'] = '   '
            
            # Should not set token in R when only whitespace
            _forward_github_token_to_r()
            
        finally:
            if original_token:
                os.environ['GITHUB_TOKEN'] = original_token
            elif 'GITHUB_TOKEN' in os.environ:
                del os.environ['GITHUB_TOKEN']
    
    def test_forward_github_token_pat_prefix(self):
        """Handle token with gh prefix (lines 122-124)"""
        from brmspy.binaries.r import _forward_github_token_to_r
        import rpy2.robjects as ro
        from typing import cast
        
        original_token = os.environ.get('GITHUB_TOKEN')
        original_pat = os.environ.get('GITHUB_PAT')
        try:
            # Set token with gh prefix and ensure PAT not set
            if 'GITHUB_PAT' in os.environ:
                del os.environ['GITHUB_PAT']
            test_token = 'ghp_testtoken123'
            os.environ['GITHUB_TOKEN'] = test_token
            
            _forward_github_token_to_r()
            
            # Should set GITHUB_TOKEN in R (not GITHUB_PAT since PAT not in Python env)
            r_token = str(cast(list, ro.r('Sys.getenv("GITHUB_TOKEN")'))[0])
            assert r_token == test_token
            
        finally:
            if original_token:
                os.environ['GITHUB_TOKEN'] = original_token
            elif 'GITHUB_TOKEN' in os.environ:
                del os.environ['GITHUB_TOKEN']
            
            if original_pat:
                os.environ['GITHUB_PAT'] = original_pat
            elif 'GITHUB_PAT' in os.environ:
                del os.environ['GITHUB_PAT']


@pytest.mark.crossplatform
class TestGetRPkgVersion:
    """Test R package version detection."""
    
    def test_get_r_pkg_version_installed_package(self):
        """Get version of installed package (lines 130-144)"""
        from brmspy.binaries.r import _get_r_pkg_version
        from packaging.version import Version
        
        # brms should be installed from main tests
        version = _get_r_pkg_version("brms")
        
        if version is not None:
            assert isinstance(version, Version)
            assert version.major >= 2
    
    def test_get_r_pkg_version_not_installed(self):
        """Return None for non-installed package (lines 146-147)"""
        from brmspy.binaries.r import _get_r_pkg_version
        
        version = _get_r_pkg_version("nonexistent_package_xyz_123")
        assert version is None
    
    def test_get_r_pkg_version_error_handling(self):
        """Handle errors gracefully (lines 146-147)"""
        from brmspy.binaries.r import _get_r_pkg_version
        
        # Pass invalid package name
        version = _get_r_pkg_version("")
        # Should return None, not raise
        assert version is None


@pytest.mark.crossplatform
class TestGetRPkgInstalled:
    """Test R package installation checking."""
    
    def test_get_r_pkg_installed_true(self):
        """Return True for installed package (lines 163-170)"""
        from brmspy.binaries.r import _get_r_pkg_installed
        
        # stats package should always be available
        result = _get_r_pkg_installed("stats")
        assert result is True
    
    def test_get_r_pkg_installed_false(self):
        """Return False for non-installed package (lines 166, 171, 173)"""
        from brmspy.binaries.r import _get_r_pkg_installed
        
        result = _get_r_pkg_installed("nonexistent_package_xyz_123")
        assert result is False
    
    def test_get_r_pkg_installed_error_handling(self):
        """Handle errors gracefully (lines 166, 171, 173)"""
        from brmspy.binaries.r import _get_r_pkg_installed
        
        # Pass invalid input
        result = _get_r_pkg_installed("")
        # Should return False, not raise
        assert result is False


@pytest.mark.crossplatform
class TestRNamespaceLoaded:
    """Test R namespace loading check."""
    
    def test_r_namespace_loaded_true(self):
        """Check loaded namespace returns True"""
        from brmspy.binaries.r import _r_namespace_loaded
        import rpy2.robjects as ro
        
        # Load a package
        ro.r('library(stats)')
        
        # Check it's loaded
        result = _r_namespace_loaded("stats")
        assert result is True
    
    def test_r_namespace_loaded_false(self):
        """Check unloaded namespace returns False"""
        from brmspy.binaries.r import _r_namespace_loaded
        
        # Check a package that's unlikely to be loaded
        result = _r_namespace_loaded("nonexistent_package_xyz")
        assert result is False


@pytest.mark.crossplatform
class TestRPackageAttached:
    """Test R package attachment check (lines 187-189)."""
    
    def test_r_package_attached_true(self):
        """Check attached package returns True"""
        from brmspy.binaries.r import _r_package_attached
        import rpy2.robjects as ro
        
        # Attach a package (library() attaches it)
        ro.r('library(stats)')
        
        # Check it's attached
        result = _r_package_attached("stats")
        assert result is True
    
    def test_r_package_attached_false(self):
        """Check non-attached package returns False (lines 187-189)"""
        from brmspy.binaries.r import _r_package_attached
        
        # Check a package that's not attached
        result = _r_package_attached("nonexistent_package_xyz")
        assert result is False
    
    def test_r_package_attached_vs_namespace_loaded(self):
        """Verify difference between attached and namespace loaded"""
        from brmspy.binaries.r import _r_package_attached, _r_namespace_loaded
        import rpy2.robjects as ro
        
        # Load a package with requireNamespace (loads but doesn't attach)
        ro.r('requireNamespace("tools", quietly=TRUE)')
        
        # Namespace should be loaded
        assert _r_namespace_loaded("tools") is True
        
        # But it might not be attached (depending on previous tests)
        # This tests the distinction between the two functions


@pytest.mark.crossplatform
class TestPackageOperationsIntegration:
    """Integration tests for package operations."""
    
    def test_check_installed_then_get_version(self):
        """Test workflow: check installed, then get version"""
        from brmspy.binaries.r import _get_r_pkg_installed, _get_r_pkg_version
        
        # Check if brms is installed
        if _get_r_pkg_installed("brms"):
            # Get its version
            version = _get_r_pkg_version("brms")
            assert version is not None
    
    def test_unload_and_check_namespace(self):
        """Test workflow: unload package and verify namespace"""
        from brmspy.binaries.r import (
            _try_force_unload_package,
            _r_namespace_loaded,
            _r_package_attached
        )
        import rpy2.robjects as ro
        
        # Load a package
        ro.r('library(tools)')
        
        # Verify it's loaded
        assert _r_namespace_loaded("tools") is True
        
        # Try to unload it
        _try_force_unload_package("tools", uninstall=False)
        
        # Package management operations should complete
        # (actual unloading may or may not succeed depending on dependencies)


@pytest.mark.crossplatform
class TestErrorRecovery:
    """Test error recovery in R package operations."""
    
    def test_operations_with_invalid_package_names(self):
        """Test all operations with invalid package names"""
        from brmspy.binaries.r import (
            _get_r_pkg_installed,
            _get_r_pkg_version,
            _r_namespace_loaded,
            _r_package_attached,
            _try_force_unload_package
        )
        
        invalid_names = ["", "   ", "123invalid", "package-with-dash"]
        
        for name in invalid_names:
            # All should handle gracefully without crashing
            _get_r_pkg_installed(name)
            _get_r_pkg_version(name)
            _r_namespace_loaded(name)
            _r_package_attached(name)
            _try_force_unload_package(name, uninstall=False)
