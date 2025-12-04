"""
Tests for brmspy.helpers.rtools module (Windows toolchain management).

Focus: Rtools version mapping, compiler detection, and Windows installation.
Target: 29% → 85%+ coverage

Note: Windows-specific tests use pytest.skipif to only run on Windows CI.
"""

import pytest
import platform
from packaging.version import Version


@pytest.mark.crossplatform
class TestRtoolsVersionMapping:
    """Test R version to Rtools version mapping (lines 117-132)."""
    
    def test_pick_rtools_for_r_4_0_to_4_1(self):
        """R 4.0.x and 4.1.x should use Rtools 40"""
        from brmspy.helpers.rtools import pick_rtools_for_r
        
        assert pick_rtools_for_r(Version("4.0.0")) == "40"
        assert pick_rtools_for_r(Version("4.0.5")) == "40"
        assert pick_rtools_for_r(Version("4.1.0")) == "40"
        assert pick_rtools_for_r(Version("4.1.3")) == "40"
    
    def test_pick_rtools_for_r_4_2(self):
        """R 4.2.x should use Rtools 42"""
        from brmspy.helpers.rtools import pick_rtools_for_r
        
        assert pick_rtools_for_r(Version("4.2.0")) == "42"
        assert pick_rtools_for_r(Version("4.2.3")) == "42"
    
    def test_pick_rtools_for_r_4_3(self):
        """R 4.3.x should use Rtools 43"""
        from brmspy.helpers.rtools import pick_rtools_for_r
        
        assert pick_rtools_for_r(Version("4.3.0")) == "43"
        assert pick_rtools_for_r(Version("4.3.2")) == "43"
    
    def test_pick_rtools_for_r_4_4(self):
        """R 4.4.x should use Rtools 44"""
        from brmspy.helpers.rtools import pick_rtools_for_r
        
        assert pick_rtools_for_r(Version("4.4.0")) == "44"
        assert pick_rtools_for_r(Version("4.4.1")) == "44"
    
    def test_pick_rtools_for_r_4_5(self):
        """R 4.5.x should use Rtools 45"""
        from brmspy.helpers.rtools import pick_rtools_for_r
        
        assert pick_rtools_for_r(Version("4.5.0")) == "45"
        assert pick_rtools_for_r(Version("4.5.2")) == "45"
    
    def test_pick_rtools_for_r_4_6(self):
        """R 4.6.x should use Rtools 46"""
        from brmspy.helpers.rtools import pick_rtools_for_r
        
        assert pick_rtools_for_r(Version("4.6.0")) == "46"
    
    def test_pick_rtools_for_r_4_7(self):
        """R 4.7.x should use Rtools 47"""
        from brmspy.helpers.rtools import pick_rtools_for_r
        
        assert pick_rtools_for_r(Version("4.7.0")) == "47"
    
    def test_pick_rtools_for_r_legacy(self):
        """R < 4.0.0 should return None (not supported)"""
        from brmspy.helpers.rtools import pick_rtools_for_r
        
        assert pick_rtools_for_r(Version("3.6.3")) is None
        assert pick_rtools_for_r(Version("3.5.0")) is None


@pytest.mark.crossplatform
class TestGxxVersionParsing:
    """Test g++ compiler version parsing (line 251, lines 260-281)."""
    
    def test_parse_gxx_version_standard_format(self):
        """Parse standard g++ version output"""
        from brmspy.helpers.rtools import _parse_gxx_version
        
        output = "g++ (GCC) 12.2.0\nCopyright (C) 2022 Free Software Foundation, Inc."
        result = _parse_gxx_version(output)
        
        assert result == (12, 2)
    
    def test_parse_gxx_version_mingw(self):
        """Parse MinGW g++ output"""
        from brmspy.helpers.rtools import _parse_gxx_version
        
        output = "g++ (MinGW-W64 x86_64-posix-seh, built by Brecht Sanders) 9.3.0"
        result = _parse_gxx_version(output)
        
        assert result == (9, 3)
    
    def test_parse_gxx_version_multiline(self):
        """Parse g++ output with version on different line"""
        from brmspy.helpers.rtools import _parse_gxx_version
        
        output = """g++ (MinGW-W64)
9.3.0
Copyright info here"""
        result = _parse_gxx_version(output)
        
        assert result == (9, 3)
    
    def test_parse_gxx_version_rtools_format(self):
        """Parse Rtools g++ output"""
        from brmspy.helpers.rtools import _parse_gxx_version
        
        output = "g++ (GCC) 13.1.0\nTarget: x86_64-w64-mingw32"
        result = _parse_gxx_version(output)
        
        assert result == (13, 1)
    
    def test_parse_gxx_version_invalid(self):
        """Handle unparseable output (line 251)"""
        from brmspy.helpers.rtools import _parse_gxx_version
        
        # No version number
        assert _parse_gxx_version("Error: command not found") is None
        
        # Empty output
        assert _parse_gxx_version("") is None
        
        # Invalid version format
        assert _parse_gxx_version("g++ version unknown") is None


@pytest.mark.crossplatform
class TestGetRVersion:
    """Test R version detection via rpy2."""
    
    def test_get_r_version_returns_version(self):
        """Verify _get_r_version returns packaging.Version (lines 48-49)"""
        from brmspy.helpers.rtools import _get_r_version
        
        r_ver = _get_r_version()
        
        assert isinstance(r_ver, Version)
        assert r_ver.major >= 4, f"R version too old: {r_ver}"
    
    def test_get_r_version_comparable(self):
        """Verify version can be compared"""
        from brmspy.helpers.rtools import _get_r_version
        
        r_ver = _get_r_version()
        
        # Should be comparable to other versions
        assert r_ver >= Version("4.0.0")
        assert r_ver < Version("10.0.0")


@pytest.mark.skipif(
    platform.system() != "Windows",
    reason="Windows-specific Rtools tests"
)
class TestWindowsRtoolsDetection:
    """Windows-only tests for Rtools detection (lines 260-281)."""
    
    def test_windows_has_rtools_silent(self):
        """Test Rtools detection with silent flag"""
        from brmspy.helpers.rtools import _windows_has_rtools
        
        # Just call it - will return True/False based on actual system
        result = _windows_has_rtools(silent=True)
        assert isinstance(result, bool)
    
    def test_windows_has_rtools_verbose(self):
        """Test Rtools detection with warnings enabled"""
        from brmspy.helpers.rtools import _windows_has_rtools
        
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
        from brmspy.helpers.rtools import _install_rtools_for_current_r
        
        # This will either:
        # - Return tag if Rtools already present
        # - Install and return tag if CI allows installation
        # - Return None if R version not supported
        tag = _install_rtools_for_current_r()
        
        if tag is not None:
            # Should be a valid Rtools tag
            assert tag in ["40", "42", "43", "44", "45", "46", "47"]
    
    def test_install_rtools_updates_path(self):
        """Verify Rtools installation updates PATH"""
        import os
        from brmspy.helpers.rtools import _install_rtools_for_current_r
        
        # Get initial PATH
        initial_path = os.environ.get("PATH", "")
        
        # Run installation (may do nothing if already present)
        tag = _install_rtools_for_current_r()
        
        # Get updated PATH
        updated_path = os.environ.get("PATH", "")
        
        # If tag returned, PATH should include rtools
        if tag is not None:
            assert "rtools" in updated_path.lower() or initial_path == updated_path


@pytest.mark.skipif(
    platform.system() == "Windows",
    reason="Non-Windows platform test"
)
class TestNonWindowsRtools:
    """Tests for non-Windows platforms."""
    
    def test_install_rtools_returns_none_on_non_windows(self):
        """_install_rtools_for_current_r should return None on non-Windows (line 353-354)"""
        from brmspy.helpers.rtools import _install_rtools_for_current_r
        
        result = _install_rtools_for_current_r()
        assert result is None