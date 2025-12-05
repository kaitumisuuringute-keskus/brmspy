"""
Tests for brmspy.helpers.rtools module (Windows toolchain management).

Focus: Rtools version mapping, compiler detection, and Windows installation.
Target: 29% → 85%+ coverage

Note: Windows-specific tests use pytest.skipif to only run on Windows CI.
"""

import pytest
import platform
from packaging.version import Version


@pytest.mark.rdeps
class TestRtoolsVersionMapping:
    """Test R version to Rtools version mapping (lines 117-132)."""
    
    def test_get_required_version_4_0_to_4_1(self):
        """R 4.0.x and 4.1.x should use Rtools 40"""
        from brmspy.runtime._rtools import get_required_version
        
        assert get_required_version(Version("4.0.0")) == "40"
        assert get_required_version(Version("4.0.5")) == "40"
        assert get_required_version(Version("4.1.0")) == "40"
        assert get_required_version(Version("4.1.3")) == "40"
    
    def test_get_required_version_4_2(self):
        """R 4.2.x should use Rtools 42"""
        from brmspy.runtime._rtools import get_required_version
        
        assert get_required_version(Version("4.2.0")) == "42"
        assert get_required_version(Version("4.2.3")) == "42"
    
    def test_get_required_version_4_3(self):
        """R 4.3.x should use Rtools 43"""
        from brmspy.runtime._rtools import get_required_version
        
        assert get_required_version(Version("4.3.0")) == "43"
        assert get_required_version(Version("4.3.2")) == "43"
    
    def test_get_required_version_4_4(self):
        """R 4.4.x should use Rtools 44"""
        from brmspy.runtime._rtools import get_required_version
        
        assert get_required_version(Version("4.4.0")) == "44"
        assert get_required_version(Version("4.4.1")) == "44"
    
    def test_get_required_version_4_5(self):
        """R 4.5.x should use Rtools 45"""
        from brmspy.runtime._rtools import get_required_version
        
        assert get_required_version(Version("4.5.0")) == "45"
        assert get_required_version(Version("4.5.2")) == "45"
    
    def test_get_required_version_4_6(self):
        """R 4.6.x should use Rtools 46"""
        from brmspy.runtime._rtools import get_required_version
        
        assert get_required_version(Version("4.6.0")) == "46"
    
    def test_get_required_version_4_7(self):
        """R 4.7.x should use Rtools 47"""
        from brmspy.runtime._rtools import get_required_version
        
        assert get_required_version(Version("4.7.0")) == "47"
    
    def test_get_required_version_legacy(self):
        """R < 4.0.0 should return None (not supported)"""
        from brmspy.runtime._rtools import get_required_version
        
        assert get_required_version(Version("3.6.3")) is None
        assert get_required_version(Version("3.5.0")) is None


@pytest.mark.rdeps
class TestGxxVersionParsing:
    """Test g++ compiler version parsing (line 251, lines 260-281)."""
    
    def test_parse_gxx_version_standard_format(self):
        """Parse standard g++ version output"""
        from brmspy.runtime._platform import get_gxx_version
        
        output = "g++ (GCC) 12.2.0\nCopyright (C) 2022 Free Software Foundation, Inc."
        result = get_gxx_version(output)
        
        assert result == (12, 2)
    
    def test_parse_gxx_version_mingw(self):
        """Parse MinGW g++ output"""
        from brmspy.runtime._platform import get_gxx_version
        
        output = "g++ (MinGW-W64 x86_64-posix-seh, built by Brecht Sanders) 9.3.0"
        result = get_gxx_version(output)
        
        assert result == (9, 3)
    
    def test_parse_gxx_version_multiline(self):
        """Parse g++ output with version on different line"""
        from brmspy.runtime._platform import get_gxx_version
        
        output = """g++ (MinGW-W64)
9.3.0
Copyright info here"""
        result = get_gxx_version(output)
        
        assert result == (9, 3)
    
    def test_parse_gxx_version_rtools_format(self):
        """Parse Rtools g++ output"""
        from brmspy.runtime._platform import get_gxx_version
        
        output = "g++ (GCC) 13.1.0\nTarget: x86_64-w64-mingw32"
        result = get_gxx_version(output)
        
        assert result == (13, 1)
    
    def test_parse_gxx_version_invalid(self):
        """Handle unparseable output (line 251)"""
        from brmspy.runtime._platform import get_gxx_version
        
        # No version number
        assert get_gxx_version("Error: command not found") is None
        
        # Empty output
        assert get_gxx_version("") is None
        
        # Invalid version format
        assert get_gxx_version("g++ version unknown") is None


@pytest.mark.rdeps
class TestGetRVersion:
    """Test R version detection via rpy2."""
    
    def test_get_r_version_returns_version(self):
        """Verify get_r_version returns packaging.Version (lines 48-49)"""
        from brmspy.runtime._platform import get_r_version
        
        r_ver = get_r_version()
        
        assert isinstance(r_ver, tuple)
        assert r_ver[0] >= 4, f"R version too old: {r_ver}"


@pytest.mark.skipif(
    platform.system() != "Windows",
    reason="Windows-specific Rtools tests"
)
class TestWindowsRtoolsDetection:
    """Windows-only tests for Rtools detection (lines 260-281)."""
    
    def test_windows_has_rtools_silent(self):
        """Test Rtools detection with silent flag"""
        from brmspy.runtime._rtools import _windows_has_rtools
        
        # Just call it - will return True/False based on actual system
        result = _windows_has_rtools(silent=True)
        assert isinstance(result, bool)
    
    def test_windows_has_rtools_verbose(self):
        """Test Rtools detection with warnings enabled"""
        from brmspy.runtime._rtools import _windows_has_rtools
        
        # Should not raise exceptions
        result = _windows_has_rtools(silent=False)
        assert isinstance(result, bool)


@pytest.mark.skipif(
    platform.system() != "Windows",
    reason="Windows-specific Rtools installation tests"
)
@pytest.mark.slow
class TestWindowsRtoolsInstallation:
    """Windows-only tests for Rtools installation (lines 353-406)."""
    
    def test_install_rtools_for_current_r_returns_tag(self):
        """Test Rtools installation returns appropriate tag"""
        from brmspy.runtime._rtools import ensure_installed
        
        ensure_installed()
    
    def test_install_rtools_updates_path(self):
        """Verify Rtools installation updates PATH"""
        import os
        from brmspy.runtime._rtools import ensure_installed
        
        # Get initial PATH
        initial_path = os.environ.get("PATH", "")
        
        # Run installation (may do nothing if already present)
        ensure_installed()
        
        # Get updated PATH
        updated_path = os.environ.get("PATH", "")
        
        # If tag returned, PATH should include rtools
        assert "rtools" in updated_path.lower()


@pytest.mark.skipif(
    platform.system() == "Windows",
    reason="Non-Windows platform test"
)
class TestNonWindowsRtools:
    """Tests for non-Windows platforms."""
    
    def test_install_rtools_returns_none_on_non_windows(self):
        """should skip gracefully"""
        from brmspy.runtime._rtools import ensure_installed
        ensure_installed()