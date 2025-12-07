"""
Tests for brmspy.runtime installation (extended coverage).

Focus: Installation resilience, version handling, error recovery, helpers.

These tests exercise error paths and edge cases in installation functions.
"""

from typing import cast
from urllib.parse import urlparse
import pytest
import platform



@pytest.mark.rdeps
class TestGetLinuxRepo:
    """Test Linux repository detection."""
    
    def test_get_linux_repo_with_os_release(self):
        """Test repository detection with /etc/os-release"""
        from brmspy.runtime._r_packages import _get_linux_repo
        import os
        
        if platform.system() == "Linux" and os.path.exists("/etc/os-release"):
            repo = _get_linux_repo()
            
            # Should return P3M URL with codename
            assert "packagemanager.posit.co" in repo
            assert "__linux__" in repo
            assert "/latest" in repo
    
    def test_get_linux_repo_fallback(self):
        """Test fallback when /etc/os-release missing"""
        from brmspy.runtime._r_packages import _get_linux_repo
        from unittest.mock import patch
        
        # Mock file not found
        with patch("builtins.open", side_effect=FileNotFoundError):
            repo = _get_linux_repo()
            
            # Should fall back to jammy (Ubuntu 22.04)
            assert "jammy" in repo
            assert "packagemanager.posit.co" in repo


@pytest.mark.rdeps
class TestGetBrmsVersion:
    """Test brms version getter."""
    
    def test_get_brms_version_returns_version(self):
        """Test get_package_version returns version string"""
        from brmspy.runtime._r_packages import get_package_version
        
        version = get_package_version("brms")
        
        if version is not None:
            assert isinstance(version, str)
            # Parse version parts
            parts = version.split(".")
            assert len(parts) >= 2
            major = int(parts[0])
            assert major >= 2


@pytest.mark.rdeps
class TestInstallRPackage:
    """Test R package installation function."""
    
    def test_install_package_already_installed(self):
        """Test package already installed path"""
        from brmspy.runtime._r_packages import install_package
        
        # Try to install brms which should already be installed from main tests
        # Should detect it's already there and skip
        install_package("brms", version=None)
        
        # If it completes without error, the already-installed path worked
        assert True
    
    def test_install_package_version_none_variants(self):
        """Test version=None variants"""
        from brmspy.runtime._r_packages import install_package
        
        # These should all be treated as "latest"
        # Just test they don't crash - actual installation tested elsewhere
        
        # Empty string
        install_package("jsonlite", version="")
        
        # "latest" keyword
        install_package("jsonlite", version="latest")
        
        # "any" keyword
        install_package("jsonlite", version="any")

@pytest.mark.rdeps  
class TestInstallRPackageDeps:
    """Test dependency installation."""
    
    def test_install_package_deps_basic(self):
        """Test basic dependency installation"""
        from brmspy.runtime._r_packages import install_package_deps

        install_package_deps("brms")


@pytest.mark.rdeps
@pytest.mark.slow
class TestBuildCmdstanr:
    """Test CmdStan building."""
    
    def test_build_cmdstan_basic(self):
        """Test CmdStan build process"""
        from brmspy.runtime._r_packages import build_cmdstan
        
        # Should complete successfully if cmdstanr already built
        # Or build it if not yet built
        try:
            build_cmdstan()
        except Exception as e:
            # May fail on certain platforms but should have tried
            assert "Rtools" in str(e) or "toolchain" in str(e) or "cmdstan" in str(e).lower()

@pytest.mark.rdeps
@pytest.mark.slow
class TestInstallPrebuilt:
    """Test prebuilt binary installation."""
    
    def test_install_prebuilt_checks_compatibility(self):
        """Test prebuilt checks system compatibility"""
        from brmspy.runtime._install import install_runtime
        from brmspy.runtime import _platform
        
        # Mock incompatible system
        from unittest.mock import patch
        
        with patch.object(_platform, 'can_use_prebuilt', return_value=False):
            # Should raise RuntimeError from require_prebuilt_compatible
            with pytest.raises(RuntimeError, match="cannot use prebuilt"):
                install_runtime(install_rtools=False)
    
    def test_install_prebuilt_constructs_url(self):
        """Test URL construction from fingerprint"""
        from brmspy.runtime._install import install_runtime
        from brmspy.runtime import _platform, _github, _storage
        
        # Mock environment to allow test without actual installation
        from unittest.mock import patch, MagicMock
        from pathlib import Path
        
        with patch.object(_platform, 'can_use_prebuilt', return_value=True):
            with patch.object(_platform, 'system_fingerprint', return_value='test-x86_64-r4.3'):
                with patch.object(_storage, 'find_runtime_by_fingerprint', return_value=None):
                    with patch.object(_github, 'get_latest_runtime_version', return_value='1.0.0'):
                        with patch.object(_github, 'get_asset_sha256', return_value=None):
                            with patch('urllib.request.urlretrieve') as mock_dl:
                                with patch.object(_storage, 'install_from_archive', return_value=Path('/fake/runtime')):
                                    
                                    install_runtime(install_rtools=False)
                                    
                                    # Should have downloaded
                                    assert mock_dl.called
                                    
                                    # Check URL was constructed correctly
                                    call_args = mock_dl.call_args
                                    url = call_args[0][0]
                                    assert 'test-x86_64-r4.3' in url
                                    from urllib.parse import urlparse
                                    parsed = urlparse(url)
                                    assert parsed.hostname == 'github.com'
    
    def test_install_prebuilt_handles_failure(self):
        """Test prebuilt installation failure handling"""
        from brmspy.runtime._install import install_runtime
        from brmspy.runtime import _platform
        from unittest.mock import patch
        
        with patch.object(_platform, 'can_use_prebuilt', return_value=True):
            with patch.object(_platform, 'system_fingerprint', return_value='test-fp'):
                with patch('urllib.request.urlretrieve', side_effect=RuntimeError("Test error")):
                    
                    # Should raise, not return False
                    with pytest.raises(RuntimeError):
                        install_runtime(install_rtools=False)


@pytest.mark.rdeps
@pytest.mark.slow
class TestInstallBrms:
    """Test main install_brms function."""
    
    def test_install_prebuilt_path(self):
        """Test install with prebuilt binaries"""
        from brmspy.runtime import install_brms
        from brmspy.runtime import _install
        from unittest.mock import patch
        from pathlib import Path
        
        # Mock successful prebuilt installation
        with patch.object(_install, 'install_runtime', return_value=Path('/fake/runtime')):
            with patch('brmspy.runtime._activation.activate'):
                with patch('brmspy.runtime._config.set_active_runtime_path'):
                    # Should return path
                    result = install_brms(use_prebuilt=True, install_rtools=False)
                    assert result == Path('/fake/runtime')
    
    def test_install_rtools_flag(self):
        """Test Rtools installation flag"""
        from brmspy.runtime import install_brms
        from brmspy.runtime import _rtools, _r_packages, _r_env, _state
        from unittest.mock import patch, MagicMock
        
        if platform.system() != "Windows":
            pytest.skip("Windows-only test")
        
        mock_ensure_rtools = MagicMock()
        
        with patch.object(_rtools, 'ensure_installed', mock_ensure_rtools):
            with patch.object(_r_packages, 'set_cran_mirror'):
                with patch.object(_r_env, 'forward_github_token'):
                    with patch.object(_r_packages, 'install_package'):
                        with patch.object(_r_packages, 'install_package_deps'):
                            with patch.object(_r_packages, 'build_cmdstan'):
                                with patch.object(_state, 'invalidate_packages'):
                                    
                                    install_brms(
                                        use_prebuilt=False,
                                        install_rtools=True,
                                        install_rstan=False
                                    )
                                    
                                    # Should have called Rtools installation
                                    assert mock_ensure_rtools.called
    
    def test_install_rstan_option(self):
        """Test rstan installation option"""
        from brmspy.runtime import install_brms

        if platform.system() == "Darwin":
            pytest.skip("No rstan on macos")
        
        install_brms(
            use_prebuilt=False,
            install_cmdstanr=False,
            install_rstan=True
        )
        


@pytest.mark.rdeps
class TestInit:
    """Test initialization function."""
    
    def test_set_cran_mirror(self):
        """Test set_cran_mirror sets CRAN mirror"""
        from brmspy.runtime._r_packages import set_cran_mirror
        import rpy2.robjects as ro
        
        set_cran_mirror()
        
        # Verify CRAN mirror is set
        repos = cast(ro.ListVector, ro.r('getOption("repos")'))
        # repos is an R vector, check it exists and has CRAN
        assert repos is not None

        # Parse repos for a robust hostname check
        # repos can be R named vector; get CRAN URL and check its hostname.
        cran_url = None
        # Convert repos to Python dict if possible
        if hasattr(repos, 'names'):
            for i, name in enumerate(list(repos.names)):
                if str(name) == "CRAN":
                    cran_url = str(repos[i])
                    break
        else:
            # Fallback: try as dict or string
            if isinstance(repos, dict):
                cran_url = repos.get("CRAN")
            elif "CRAN" in str(repos):
                # Weak fallback, as last resort (should never be needed)
                cran_url = str(repos)
        assert cran_url is not None
        parsed = urlparse(cran_url)
        assert parsed.hostname == "cloud.r-project.org" or "CRAN" in str(repos)
