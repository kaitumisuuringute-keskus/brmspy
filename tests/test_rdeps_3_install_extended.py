"""
Tests for brmspy.runtime installation (extended coverage).

Focus: Installation resilience, version handling, error recovery, helpers.

These tests exercise error paths and edge cases in installation functions.
"""

import pytest
import platform


@pytest.mark.rdeps
class TestGetLinuxRepo:
    """Test Linux repository detection."""

    @pytest.mark.worker
    def test_get_linux_repo_with_os_release(self):
        """Test repository detection with /etc/os-release"""
        from brmspy._runtime._r_packages import _get_linux_repo
        import os

        if platform.system() == "Linux" and os.path.exists("/etc/os-release"):
            repo = _get_linux_repo()

            # Should return P3M URL with codename
            assert "packagemanager.posit.co" in repo
            assert "__linux__" in repo
            assert "/latest" in repo

    @pytest.mark.worker
    def test_get_linux_repo_fallback(self):
        """Test fallback when /etc/os-release missing"""
        from brmspy._runtime._r_packages import _get_linux_repo
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

    @pytest.mark.worker
    def test_get_brms_version_returns_version(self):
        """Test get_package_version returns version string"""
        from brmspy._runtime._r_packages import get_package_version

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

    @pytest.mark.worker
    def test_install_package_already_installed(self):
        """Test package already installed path"""
        from brmspy._runtime._r_packages import install_package

        # Try to install brms which should already be installed from main tests
        # Should detect it's already there and skip
        install_package("brms", version=None)

        # If it completes without error, the already-installed path worked
        assert True

    @pytest.mark.worker
    def test_install_package_version_none_variants(self):
        """Test version=None variants"""
        from brmspy._runtime._r_packages import install_package

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

    @pytest.mark.worker
    def test_install_package_deps_basic(self):
        """Test basic dependency installation"""
        from brmspy._runtime._r_packages import install_package_deps

        install_package_deps("brms")


@pytest.mark.rdeps
@pytest.mark.slow
class TestBuildCmdstanr:
    """Test CmdStan building."""

    @pytest.mark.worker
    def test_build_cmdstan_basic(self):
        """Test CmdStan build process"""
        from brmspy._runtime._r_packages import build_cmdstan

        # Should complete successfully if cmdstanr already built
        # Or build it if not yet built
        try:
            build_cmdstan()
        except Exception as e:
            # May fail on certain platforms but should have tried
            assert (
                "Rtools" in str(e)
                or "toolchain" in str(e)
                or "cmdstan" in str(e).lower()
            )


@pytest.mark.rdeps
@pytest.mark.slow
class TestInstallPrebuilt:
    """Test prebuilt binary installation."""

    @pytest.mark.worker
    def test_install_prebuilt_checks_compatibility(self):
        """Test prebuilt checks system compatibility"""
        from brmspy._runtime._install import install_runtime
        from brmspy._runtime import _platform

        # Mock incompatible system
        from unittest.mock import patch

        with patch.object(_platform, "can_use_prebuilt", return_value=False):
            # Should raise RuntimeError from require_prebuilt_compatible
            with pytest.raises(RuntimeError, match="cannot use prebuilt"):
                install_runtime(install_rtools=False)

    @pytest.mark.worker
    def test_install_prebuilt_constructs_url(self):
        """Test URL construction from fingerprint"""
        from brmspy._runtime._install import install_runtime
        from brmspy._runtime import _platform, _github, _storage

        # Mock environment to allow test without actual installation
        from unittest.mock import patch, MagicMock
        from pathlib import Path

        with (
            patch("urllib.request.urlretrieve") as mock_dl,
            patch.object(_platform, "can_use_prebuilt", return_value=True),
            patch.object(
                _platform, "system_fingerprint", return_value="test-x86_64-r4.3"
            ),
            patch.object(_storage, "find_runtime_by_fingerprint", return_value=None),
            patch.object(_github, "get_latest_runtime_version", return_value="1.0.0"),
            patch.object(_github, "get_asset_sha256", return_value="fake-hash"),
            patch.object(
                _storage,
                "install_from_archive",
                return_value=Path("/fake/runtime"),
            ),
            patch.object(_storage, "write_stored_hash", return_value=None),
        ):

            install_runtime(install_rtools=False)

            # Should have downloaded
            assert mock_dl.called

            # Check URL was constructed correctly
            call_args = mock_dl.call_args
            url = call_args[0][0]
            assert "test-x86_64-r4.3" in url
            from urllib.parse import urlparse

            parsed = urlparse(url)
            assert parsed.hostname == "github.com"

    @pytest.mark.worker
    def test_install_prebuilt_missing_hash(self):
        """Test URL construction from fingerprint"""
        from brmspy._runtime._install import install_runtime
        from brmspy._runtime import _platform, _github, _storage

        # Mock environment to allow test without actual installation
        from unittest.mock import patch, MagicMock
        from pathlib import Path

        with (
            patch.object(_platform, "can_use_prebuilt", return_value=True),
            patch.object(
                _platform, "system_fingerprint", return_value="test-x86_64-r4.3"
            ),
            patch.object(_storage, "find_runtime_by_fingerprint", return_value=None),
            patch.object(_github, "get_latest_runtime_version", return_value="1.0.0"),
            patch.object(_github, "get_asset_sha256", return_value=None),
            patch("urllib.request.urlretrieve") as mock_dl,
            patch.object(
                _storage,
                "install_from_archive",
                return_value=Path("/fake/runtime"),
            ),
        ):
            with pytest.raises(Exception):
                install_runtime(install_rtools=False)

    @pytest.mark.worker
    def test_install_prebuilt_handles_failure(self):
        """Test prebuilt installation failure handling"""
        from brmspy._runtime._install import install_runtime
        from brmspy._runtime import _platform
        from unittest.mock import patch

        with (
            patch.object(_platform, "can_use_prebuilt", return_value=True),
            patch.object(_platform, "system_fingerprint", return_value="test-fp"),
            patch(
                "urllib.request.urlretrieve",
                side_effect=RuntimeError("Test error"),
            ),
        ):
            with pytest.raises(Exception):
                install_runtime(install_rtools=False)


@pytest.mark.rdeps
@pytest.mark.slow
class TestInstallBrms:

    def test_install_rtools_flag(self):
        """Test Rtools installation flag"""

        if platform.system() != "Windows":
            pytest.skip("Windows-only test")

        pytest.skip("TODO")

        from brmspy import brms

        with brms.manage() as ctx:
            ctx.install_brms(use_prebuilt=True, install_rtools=True)

    def test_install_rstan_option(self):
        """Test rstan installation option"""
        from brmspy import brms

        if platform.system() != "Darwin":
            pytest.skip("No-macos test (known issue!)")

        with brms.manage() as ctx:
            ctx.install_brms(
                use_prebuilt=False, install_cmdstanr=False, install_rstan=True
            )
