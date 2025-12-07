"""
System and environment detection. All functions are pure.
No side effects, no R environment mutation.
"""

import platform
import subprocess
from typing import cast
from brmspy.runtime._types import SystemInfo


# === Detection (pure) ===

def get_os() -> str:
    """Returns 'linux', 'macos', or 'windows'."""
    raw_os = platform.system().lower()
    if raw_os == "darwin":
        return "macos"
    elif raw_os in ("windows", "linux"):
        return raw_os
    else:
        return raw_os


def get_arch() -> str:
    """Returns 'x86_64' or 'arm64'."""
    raw_arch = platform.machine().lower()
    if raw_arch in ("x86_64", "amd64"):
        return "x86_64"
    elif raw_arch in ("arm64", "aarch64"):
        return "arm64"
    else:
        return raw_arch


def get_r_version() -> tuple[int, int, int] | None:
    """Returns R version tuple or None if R not available."""
    try:
        import rpy2.robjects as ro
        r_major = cast(ro.ListVector, ro.r("R.Version()$major"))
        r_minor = cast(ro.ListVector, ro.r("R.Version()$minor"))
        major = int(r_major[0])
        minor_str = str(r_minor[0])
        parts = minor_str.split(".")
        minor = int(parts[0]) if parts and parts[0].isdigit() else 0
        patch = int(parts[1]) if len(parts) > 1 and parts[1].isdigit() else 0
        return major, minor, patch
    except Exception:
        return None


def system_fingerprint() -> str:
    """Returns '{os}-{arch}-r{major}.{minor}' string."""
    os_name = get_os()
    arch = get_arch()
    r_ver = get_r_version()
    if not r_ver:
        raise RuntimeError("R version could not be determined")
    major, minor, _ = r_ver
    return f"{os_name}-{arch}-r{major}.{minor}"


def get_glibc_version() -> tuple[int, int] | None:
    """Linux only. Parse from ldd --version."""
    if get_os() != "linux":
        return None

    try:
        out = subprocess.check_output(["ldd", "--version"], text=True, timeout=10)
        for line in out.splitlines():
            for token in line.split():
                if token[0].isdigit() and "." in token:
                    parts = token.split(".")
                    if len(parts) >= 2 and parts[0].isdigit() and parts[1].isdigit():
                        return int(parts[0]), int(parts[1])
    except Exception:
        pass
    return None


def get_clang_version() -> tuple[int, int] | None:
    """macOS only. Parse from clang --version."""
    if get_os() != "macos":
        return None

    try:
        out = subprocess.check_output(["clang", "--version"], text=True, timeout=10)
        for line in out.splitlines():
            if "clang" not in line.lower():
                continue
            for token in line.split():
                if token[0].isdigit() and "." in token:
                    parts = token.split(".")
                    if len(parts) >= 2 and parts[0].isdigit() and parts[1].isdigit():
                        return int(parts[0]), int(parts[1])
    except Exception:
        pass
    return None


def get_gxx_version(out: None | str = None) -> tuple[int, int] | None:
    """Parse from g++ --version."""
    try:
        if out is None:
            out = subprocess.check_output(["g++", "--version"], text=True, timeout=10)
        # Parse g++ version (same logic as helpers/rtools.py)
        for line in out.splitlines():
            for token in line.split():
                if token[0].isdigit() and "." in token:
                    parts = token.split(".")
                    if len(parts) >= 2 and parts[0].isdigit() and parts[1].isdigit():
                        return int(parts[0]), int(parts[1])
    except Exception:
        pass
    return None


def get_system_info() -> SystemInfo:
    """Collect all system info into immutable dataclass."""
    os_name = get_os()
    arch = get_arch()
    r_ver = get_r_version()
    fp = system_fingerprint() if r_ver else f"{os_name}-{arch}-rUNK"
    
    # Collect toolchain info based on OS
    glibc_ver = None
    clang_ver = None
    gxx_ver = None
    has_rtools = False
    
    if os_name == "linux":
        glibc_ver = get_glibc_version()
        gxx_ver = get_gxx_version()
    elif os_name == "macos":
        clang_ver = get_clang_version()
    elif os_name == "windows":
        from ._rtools import _windows_has_rtools
        has_rtools = _windows_has_rtools(silent=True)
        gxx_ver = get_gxx_version()
    
    return SystemInfo(
        os=os_name,
        arch=arch,
        r_version=r_ver,
        fingerprint=fp,
        glibc_version=glibc_ver,
        clang_version=clang_ver,
        gxx_version=gxx_ver,
        has_rtools=has_rtools,
    )


# === Boolean queries (for status, no exceptions) ===

def is_platform_supported() -> bool:
    """True if OS/arch combination is supported."""
    os_name = get_os()
    arch = get_arch()
    supported_os = ("linux", "macos", "windows")
    supported_os_arch = {
        "windows": {"x86_64"},
        "linux": {"x86_64"},
        "macos": {"arm64"}
    }
    
    if os_name not in supported_os:
        return False
    
    supported_archs = supported_os_arch.get(os_name, set())
    if arch not in supported_archs:
        return False
    
    return True


def is_r_available() -> bool:
    """True if R is installed."""
    return get_r_version() is not None


def is_r_supported(min_version: tuple[int, int] = (4, 0)) -> bool:
    """True if R version meets minimum."""
    v = get_r_version()
    if v is None:
        return False
    major, minor, _ = v
    min_major, min_minor = min_version
    if major < min_major:
        return False
    if major == min_major and minor < min_minor:
        return False
    return True


def is_linux_toolchain_ok() -> bool:
    """True if glibc >= 2.27 and g++ >= 9."""
    glibc = get_glibc_version()
    if glibc is None or glibc < (2, 27):
        return False
    
    gxx = get_gxx_version()
    if gxx is None or gxx < (9, 0):
        return False
    
    return True


def is_macos_toolchain_ok() -> bool:
    """True if Xcode CLT installed and clang >= 11."""
    try:
        subprocess.check_output(["xcode-select", "-p"], text=True, timeout=10)
    except Exception:
        return False
    
    clang = get_clang_version()
    if clang is None or clang < (11, 0):
        return False
    
    return True


def is_windows_toolchain_ok() -> bool:
    """True if Rtools with g++ >= 9."""
    from ._rtools import _windows_has_rtools
    if not _windows_has_rtools(silent=True):
        return False
    gxx_version = get_gxx_version()
    if gxx_version is None or gxx_version < (9, 0):
        return False
    return True


def is_toolchain_compatible() -> bool:
    """Route to OS-specific check."""
    os_name = get_os()
    if os_name == "linux":
        return is_linux_toolchain_ok()
    elif os_name == "macos":
        return is_macos_toolchain_ok()
    elif os_name == "windows":
        return is_windows_toolchain_ok()
    return False


def can_use_prebuilt() -> bool:
    """Master check: platform + R + toolchain."""
    if not is_platform_supported():
        return False
    if not is_r_supported():
        return False
    if not is_toolchain_compatible():
        return False
    return True


# Prebuilt fingerprints set (currently empty, can be populated)
PREBUILT_FINGERPRINTS: set[str] = set()


def is_prebuilt_available(fingerprint: str) -> bool:
    """True if we publish prebuilt for this fingerprint."""
    # Currently we allow all fingerprints if can_use_prebuilt() passes
    # In future, this could check against PREBUILT_FINGERPRINTS
    return fingerprint in PREBUILT_FINGERPRINTS if PREBUILT_FINGERPRINTS else True


def get_compatibility_issues() -> list[str]:
    """Return list of human-readable compatibility issues."""
    issues = []
    
    if not is_platform_supported():
        os_name = get_os()
        arch = get_arch()
        issues.append(f"Unsupported platform: {os_name}-{arch}")
    
    if not is_r_available():
        issues.append("R is not available")
    elif not is_r_supported():
        r_ver = get_r_version()
        issues.append(f"R version {r_ver} is too old (need >= 4.0)")
    
    if not is_toolchain_compatible():
        os_name = get_os()
        if os_name == "linux":
            glibc = get_glibc_version()
            gxx = get_gxx_version()
            if glibc is None or glibc < (2, 27):
                issues.append(f"glibc {glibc} is too old (need >= 2.27)")
            if gxx is None or gxx < (9, 0):
                issues.append(f"g++ {gxx} is too old (need >= 9.0)")
        elif os_name == "macos":
            try:
                subprocess.check_output(["xcode-select", "-p"], text=True, timeout=10)
            except Exception:
                issues.append("Xcode Command Line Tools not installed")
            clang = get_clang_version()
            if clang is None or clang < (11, 0):
                issues.append(f"clang {clang} is too old (need >= 11.0)")
        elif os_name == "windows":
            if not is_windows_toolchain_ok():
                issues.append("Rtools not found or insufficient version")
    
    return issues


# === Guard functions (raise on failure) ===

def require_platform_supported() -> None:
    """Raise RuntimeError with actionable message if unsupported."""
    if not is_platform_supported():
        os_name = get_os()
        arch = get_arch()
        raise RuntimeError(
            f"Platform {os_name}-{arch} is not supported. "
            f"Supported platforms: linux-x86_64, macos-arm64, windows-x86_64"
        )


def require_r_available(min_version: tuple[int, int] = (4, 0)) -> None:
    """Raise RuntimeError if R missing or too old."""
    if not is_r_available():
        raise RuntimeError(
            "R is not available. Please install R >= 4.0 from https://cran.r-project.org/"
        )
    if not is_r_supported(min_version):
        r_ver = get_r_version()
        raise RuntimeError(
            f"R version {r_ver} is too old. Please upgrade to R >= {min_version[0]}.{min_version[1]}"
        )


def require_toolchain_compatible() -> None:
    """Raise RuntimeError with install instructions if toolchain insufficient."""
    os_name = get_os()
    
    if os_name == "linux":
        if not is_linux_toolchain_ok():
            glibc = get_glibc_version()
            gxx = get_gxx_version()
            raise RuntimeError(
                f"Linux toolchain insufficient:\n"
                f"  glibc: {glibc} (need >= 2.27)\n"
                f"  g++: {gxx} (need >= 9.0)\n"
                f"Please upgrade your system or install from source."
            )
    elif os_name == "macos":
        if not is_macos_toolchain_ok():
            raise RuntimeError(
                "macOS toolchain insufficient:\n"
                "  Install Xcode Command Line Tools: xcode-select --install\n"
                "  Need clang >= 11.0"
            )
    elif os_name == "windows":
        if not is_windows_toolchain_ok():
            raise RuntimeError(
                "Windows toolchain insufficient:\n"
                "  Install Rtools from https://cran.r-project.org/bin/windows/Rtools/\n"
                "  Or run install(install_rtools=True)"
            )


def require_prebuilt_compatible() -> None:
    """Raise RuntimeError if system can't use prebuilt runtimes."""
    issues = get_compatibility_issues()
    can_use = can_use_prebuilt()
    if issues or not can_use:
        raise RuntimeError(
            "System cannot use prebuilt runtimes:\n" +
            "\n".join(f"  - {issue}" for issue in issues)
        )