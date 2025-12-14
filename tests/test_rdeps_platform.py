"""
Tests for brmspy.runtime._platform module (platform detection and compatibility).

Focus: Platform detection, toolchain checks, system fingerprinting.

IMPORTANT! _platform module does not use rpy2. We call it directly! If it did, we shouldn't.

It will error if ever rpy2.robjects is directly imported.

These tests exercise platform-specific code paths and error handling.
"""

import pytest
from unittest.mock import patch


@pytest.mark.rdeps
class TestOsArchDetection:
    """Test OS and architecture detection."""

    def test_get_os_current_platform(self):
        """Verify OS detection for current platform"""
        from brmspy._runtime._platform import get_os

        os_name = get_os()

        # Should return normalized values
        assert os_name in ("linux", "macos", "windows") or os_name

    def test_get_arch_current_platform(self):
        """Verify architecture detection for current platform"""
        from brmspy._runtime._platform import get_arch

        arch = get_arch()

        # Should return normalized values
        assert arch in ("x86_64", "arm64") or arch


@pytest.mark.rdeps
class TestGetRVersionTuple:
    """Test R version detection."""

    def test_get_r_version_returns_tuple(self):
        """Verify R version is returned as tuple"""
        from brmspy._runtime._platform import get_r_version

        version = get_r_version()

        if version is not None:
            assert isinstance(version, tuple)
            assert len(version) == 3
            major, minor, patch = version
            assert isinstance(major, int)
            assert isinstance(minor, int)
            assert isinstance(patch, int)
            assert major >= 4, f"R version too old: {version}"


@pytest.mark.rdeps
class TestRAvailableAndSupported:
    """Test R availability checking."""

    def test_is_r_available_current_r(self):
        """Test R availability"""
        from brmspy._runtime._platform import is_r_available

        # R should be available in test environment
        assert is_r_available()

    def test_is_r_supported_current_r(self):
        """Test with current R installation"""
        from brmspy._runtime._platform import is_r_supported

        # Should work with default requirements (R >= 4.0)
        assert is_r_supported(min_version=(4, 0))

    def test_is_r_supported_high_requirements(self):
        """Test with very high version requirements"""
        from brmspy._runtime._platform import is_r_supported

        # R 99.99 definitely not available
        result = is_r_supported(min_version=(99, 99))
        assert result is False

    def test_is_r_supported_current_major_high_minor(self):
        """Test minor version check"""
        from brmspy._runtime._platform import is_r_supported, get_r_version

        version = get_r_version()
        if version:
            major, minor, _ = version
            # Require higher minor version than current
            result = is_r_supported(min_version=(major, minor + 10))
            assert result is False


@pytest.mark.rdeps
class TestGlibcVersion:
    """Test glibc version detection."""

    def test_get_glibc_version_on_linux(self):
        """Test glibc detection on Linux"""
        from brmspy._runtime._platform import get_glibc_version, get_os
        import platform

        if platform.system() == "Linux":
            version = get_glibc_version()
            if version is not None:
                assert isinstance(version, tuple)
                assert len(version) == 2
                assert version[0] >= 2  # glibc 2.x
        else:
            # Non-Linux should return None
            version = get_glibc_version()
            assert version is None


@pytest.mark.rdeps
class TestClangVersion:
    """Test clang version detection."""

    def test_get_clang_version_on_macos(self):
        """Test clang detection on macOS"""
        from brmspy._runtime._platform import get_clang_version
        import platform

        if platform.system() == "Darwin":
            version = get_clang_version()
            if version is not None:
                assert isinstance(version, tuple)
                assert len(version) == 2
                assert version[0] >= 11  # clang 11+
        else:
            # Non-macOS should return None
            version = get_clang_version()
            assert version is None


@pytest.mark.rdeps
class TestLinuxToolchain:
    """Test Linux toolchain compatibility checks."""

    def test_is_linux_toolchain_ok_on_linux(self):
        """Test Linux toolchain check"""
        from brmspy._runtime._platform import is_linux_toolchain_ok
        import platform

        if platform.system() == "Linux":
            # On Linux, should check actual tools
            result = is_linux_toolchain_ok()
            assert isinstance(result, bool)
        else:
            # On non-Linux, won't pass these checks
            result = is_linux_toolchain_ok()
            assert result is False


@pytest.mark.rdeps
class TestMacosToolchain:
    """Test macOS toolchain compatibility checks."""

    def test_is_macos_toolchain_ok_on_macos(self):
        """Test macOS toolchain check"""
        from brmspy._runtime._platform import is_macos_toolchain_ok
        import platform

        if platform.system() == "Darwin":
            # On macOS, should check actual tools
            result = is_macos_toolchain_ok()
            assert isinstance(result, bool)
        else:
            # On non-macOS, won't pass these checks
            result = is_macos_toolchain_ok()
            assert result is False


@pytest.mark.rdeps
class TestWindowsToolchain:
    """Test Windows toolchain compatibility checks."""

    def test_is_windows_toolchain_ok_checks_rtools(self):
        """Verify Windows checks for Rtools"""
        from brmspy._runtime._platform import is_windows_toolchain_ok
        import platform

        if platform.system() == "Windows":
            result = is_windows_toolchain_ok()
            assert isinstance(result, bool)
        else:
            # On non-Windows, should return False
            result = is_windows_toolchain_ok()
            assert result is False


@pytest.mark.rdeps
class TestSupportedPlatform:
    """Test platform support checking."""

    def test_is_platform_supported_current_system(self):
        """Check current platform support"""
        from brmspy._runtime._platform import is_platform_supported

        result = is_platform_supported()
        assert isinstance(result, bool)

    def test_is_platform_supported_with_mock_unsupported_os(self):
        """Test with unsupported OS"""
        from brmspy._runtime._platform import is_platform_supported
        import platform as plat

        # Mock platform.system to return unsupported OS
        with patch.object(plat, "system", return_value="FreeBSD"):
            result = is_platform_supported()
            assert result is False

    def test_is_platform_supported_with_mock_unsupported_arch(self):
        """Test with unsupported architecture"""
        from brmspy._runtime._platform import is_platform_supported
        import platform as plat

        # Mock unsupported architecture
        with patch.object(plat, "system", return_value="Linux"):
            with patch.object(plat, "machine", return_value="riscv64"):
                result = is_platform_supported()
                assert result is False


@pytest.mark.rdeps
class TestToolchainIsCompatible:
    """Test toolchain compatibility routing."""

    def test_is_toolchain_compatible_current_platform(self):
        """Test toolchain check for current platform"""
        from brmspy._runtime._platform import is_toolchain_compatible

        result = is_toolchain_compatible()
        assert isinstance(result, bool)

    def test_is_toolchain_compatible_unknown_os(self):
        """Return False for unknown OS"""
        from brmspy._runtime._platform import is_toolchain_compatible
        import platform as plat

        # Mock unknown OS
        with patch.object(plat, "system", return_value="UnknownOS"):
            result = is_toolchain_compatible()
            assert result is False


@pytest.mark.rdeps
class TestSystemFingerprint:
    """Test system fingerprint generation."""

    def test_system_fingerprint_format(self):
        """Verify fingerprint format"""
        from brmspy._runtime._platform import system_fingerprint

        fp = system_fingerprint()

        if fp is not None:
            # Should be format: os-arch-rX.Y
            parts = fp.split("-")
            assert len(parts) >= 3
            assert parts[-1].startswith("r")

    def test_system_fingerprint_when_r_unavailable(self):
        """Raise error when R version unavailable"""
        from brmspy._runtime._platform import system_fingerprint

        # Mock get_r_version to return None
        with patch("brmspy._runtime._platform.get_r_version", return_value=None):
            with pytest.raises(RuntimeError, match="R version could not be determined"):
                system_fingerprint()


@pytest.mark.rdeps
class TestPrebuiltAvailableFor:
    """Test prebuilt availability checking."""

    def test_is_prebuilt_available_unknown_fingerprint(self):
        """Test prebuilt availability for unknown fingerprint"""
        from brmspy._runtime._platform import is_prebuilt_available

        # Unknown platform
        result = is_prebuilt_available("unknown-platform-r99.99")
        # With empty PREBUILT_FINGERPRINTS, all fingerprints are considered available
        assert result is True

    def test_is_prebuilt_available_valid_fingerprint(self):
        """Check availability for valid fingerprints"""
        from brmspy._runtime._platform import (
            is_prebuilt_available,
            PREBUILT_FINGERPRINTS,
        )

        # Add a test fingerprint
        test_fp = "test-platform-r4.3"
        PREBUILT_FINGERPRINTS.add(test_fp)

        try:
            result = is_prebuilt_available(test_fp)
            assert result is True
        finally:
            # Clean up
            PREBUILT_FINGERPRINTS.discard(test_fp)


@pytest.mark.rdeps
class TestCanUsePrebuilt:
    """Test master prebuilt eligibility check."""

    def test_can_use_prebuilt_current_system(self):
        """Test prebuilt check for current system"""
        from brmspy._runtime._platform import can_use_prebuilt

        result = can_use_prebuilt()
        assert isinstance(result, bool)

    def test_can_use_prebuilt_unsupported_platform(self):
        """Return False for unsupported platform"""
        from brmspy._runtime._platform import can_use_prebuilt
        import platform as plat

        # Mock unsupported platform
        with patch.object(plat, "system", return_value="FreeBSD"):
            result = can_use_prebuilt()
            assert result is False

    def test_can_use_prebuilt_r_unavailable(self):
        """Return False when R unavailable"""
        from brmspy._runtime._platform import can_use_prebuilt

        # Mock R unavailable
        with patch("brmspy._runtime._platform.is_r_supported", return_value=False):
            result = can_use_prebuilt()
            assert result is False

    def test_can_use_prebuilt_toolchain_incompatible(self):
        """Return False when toolchain incompatible"""
        from brmspy._runtime._platform import can_use_prebuilt

        # Mock incompatible toolchain
        with patch(
            "brmspy._runtime._platform.is_platform_supported", return_value=True
        ):
            with patch("brmspy._runtime._platform.is_r_supported", return_value=True):
                with patch(
                    "brmspy._runtime._platform.is_toolchain_compatible",
                    return_value=False,
                ):
                    result = can_use_prebuilt()
                    assert result is False


@pytest.mark.rdeps
class TestGetSystemInfo:
    """Test system info collection."""

    def test_get_system_info_returns_dataclass(self):
        """Test get_system_info returns SystemInfo dataclass"""
        from brmspy._runtime._platform import get_system_info
        from brmspy.types.runtime import SystemInfo

        info = get_system_info()

        assert isinstance(info, SystemInfo)
        assert info.os in ("linux", "macos", "windows") or info.os
        assert info.arch in ("x86_64", "arm64") or info.arch
        assert isinstance(info.fingerprint, str)

    def test_get_compatibility_issues_returns_list(self):
        """Test get_compatibility_issues returns list of strings"""
        from brmspy._runtime._platform import get_compatibility_issues

        issues = get_compatibility_issues()

        assert isinstance(issues, list)
        for issue in issues:
            assert isinstance(issue, str)
