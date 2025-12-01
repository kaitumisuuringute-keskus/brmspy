import platform
import subprocess
from typing import Optional, Tuple, Set, cast

from brmspy.helpers.log import log_warning
from brmspy.helpers.rtools import _install_rtools_for_current_r, _parse_gxx_version, _windows_has_rtools


# ----- Helpers: OS / arch -----

def _normalized_os_arch() -> Tuple[str, str]:
    """
    Get normalized OS name and CPU architecture for platform identification.
    
    Converts platform-specific OS and architecture names to standardized
    values used throughout brmspy for runtime selection and fingerprinting.
    
    Returns
    -------
    tuple of (str, str)
        (os_name, arch) where:
        - os_name: 'linux', 'macos', 'windows', or raw platform.system()
        - arch: 'x86_64', 'arm64', or raw platform.machine()
    
    Notes
    -----
    Normalizations performed:
    - 'darwin' → 'macos'
    - 'amd64' → 'x86_64'
    - 'aarch64' → 'arm64'
    
    Examples
    --------
    ```python
    os_name, arch = _normalized_os_arch()
    print(f"{os_name}-{arch}")
    # Linux: "linux-x86_64"
    # macOS Intel: "macos-x86_64"
    # macOS ARM: "macos-arm64"
    # Windows: "windows-x86_64"
    ```
    """
    raw_os = platform.system().lower()
    raw_arch = platform.machine().lower()

    if raw_os == "darwin":
        os_name = "macos"
    elif raw_os == "windows":
        os_name = "windows"
    elif raw_os == "linux":
        os_name = "linux"
    else:
        os_name = raw_os  # unsupported / unknown

    if raw_arch in ("x86_64", "amd64"):
        arch = "x86_64"
    elif raw_arch in ("arm64", "aarch64"):
        arch = "arm64"
    else:
        arch = raw_arch

    return os_name, arch


# ----- Helpers: R version -----

def get_r_version_tuple() -> Optional[Tuple[int, int, int]]:
    """
    Get R version as tuple of integers.
    
    Queries the active R installation via rpy2 to retrieve version
    information. Returns None if R is not available or rpy2 cannot
    communicate with R.
    
    Returns
    -------
    tuple of (int, int, int) or None
        (major, minor, patch) version numbers, or None if unavailable
    
    Examples
    --------

    ```python
    version = get_r_version_tuple()
    if version:
        major, minor, patch = version
        print(f"R {major}.{minor}.{patch}")
    else:
        print("R not available")
    ```

    See Also
    --------
    r_available_and_supported : Check if R meets minimum requirements
    """
    try:
        import rpy2.robjects as ro
    except Exception:
        return None

    try:
        major = int(cast(list, ro.r("R.Version()$major"))[0])
        minor_str = str(cast(list, ro.r("R.Version()$minor"))[0])
        parts = minor_str.split(".")
        minor = int(parts[0]) if parts and parts[0].isdigit() else 0
        patch = int(parts[1]) if len(parts) > 1 and parts[1].isdigit() else 0
        return major, minor, patch
    except Exception:
        return None


def r_available_and_supported(min_major: int = 4, min_minor: int = 0) -> bool:
    """
    Check if R is available and meets minimum version requirements.
    
    Parameters
    ----------
    min_major : int, default=4
        Minimum required R major version
    min_minor : int, default=2
        Minimum required R minor version
    
    Returns
    -------
    bool
        True if R is available and >= min_major.min_minor
    
    Examples
    --------

    ```python
    if r_available_and_supported():
        print("R 4.2+ is available")
    
    # Check for R 4.3+
    if r_available_and_supported(min_major=4, min_minor=3):
        print("R 4.3+ is available")
    ```

    See Also
    --------
    get_r_version_tuple : Get detailed R version
    """
    v = get_r_version_tuple()
    if v is None:
        return False

    major, minor, _ = v
    if major < min_major:
        return False
    if major == min_major and minor < min_minor:
        return False

    # If rpy2 imported and R.Version() worked, libR is there and usable.
    return True


# ----- Helpers: parsing tool output -----

def extract_glibc_version(ldd_output: str) -> Optional[Tuple[int, int]]:
    """
    Parse glibc version from ldd command output.
    
    Extracts glibc version number from the output of `ldd --version`.
    Used on Linux to verify minimum glibc requirement for prebuilt binaries.
    
    Parameters
    ----------
    ldd_output : str
        Output from `ldd --version` command
    
    Returns
    -------
    tuple of (int, int) or None
        (major, minor) version, or None if not found
    
    Examples
    --------

    ```python
    import subprocess
    
    out = subprocess.check_output(["ldd", "--version"], text=True)
    version = extract_glibc_version(out)
    if version and version >= (2, 27):
        print("glibc 2.27+ available")
    ```

    See Also
    --------
    linux_can_use_prebuilt : Uses this for Linux compatibility check
    """
    for line in ldd_output.splitlines():
        for token in line.split():
            if token[0].isdigit() and "." in token:
                parts = token.split(".")
                if len(parts) >= 2 and parts[0].isdigit() and parts[1].isdigit():
                    return int(parts[0]), int(parts[1])
    return None




def parse_clang_version(version_output: str) -> Optional[Tuple[int, int]]:
    """
    Parse clang compiler version from command output.
    
    Extracts clang version number from the output of `clang --version`.
    Used on macOS to verify minimum compiler requirement.
    
    Parameters
    ----------
    version_output : str
        Output from `clang --version` command
    
    Returns
    -------
    tuple of (int, int) or None
        (major, minor) version, or None if parsing fails
    
    Examples
    --------

    ```python
    import subprocess
    
    out = subprocess.check_output(["clang", "--version"], text=True)
    version = parse_clang_version(out)
    if version and version >= (11, 0):
        print("clang 11+ available")
    ```

    See Also
    --------
    macos_can_use_prebuilt : macOS compatibility check
    """
    for line in version_output.splitlines():
        if "clang" not in line.lower():
            continue
        for token in line.split():
            if token[0].isdigit() and "." in token:
                parts = token.split(".")
                if len(parts) >= 2 and parts[0].isdigit() and parts[1].isdigit():
                    return int(parts[0]), int(parts[1])
    return None


# ----- Per-OS toolchain checks -----

def linux_can_use_prebuilt() -> bool:
    """
    Check if Linux system meets prebuilt binary requirements.
    
    Verifies that the Linux system has compatible toolchain versions
    required to use prebuilt brms runtime bundles. Checks glibc and
    g++ compiler versions.
    
    Returns
    -------
    bool
        True if system meets all requirements:
        - glibc >= 2.27 (for binary compatibility)
        - g++ >= 9.0 (for C++17 Stan compilation)
    
    Notes
    -----
    Requirements for prebuilt Linux bundles:
    - glibc 2.27+: Ensures binary compatibility with precompiled Stan models
    - g++ 9+: Minimum compiler for C++17 features used by CmdStan
    
    Examples
    --------

    ```python
    from brmspy.binaries.env import linux_can_use_prebuilt
    
    if linux_can_use_prebuilt():
        print("System can use prebuilt Linux binaries")
    else:
        print("Must build from source")
    ```

    See Also
    --------
    extract_glibc_version : Parse glibc version from ldd
    _parse_gxx_version : Parse g++ version
    toolchain_is_compatible : Master toolchain check for all platforms
    """
    try:
        out = subprocess.check_output(["ldd", "--version"], text=True)
        glibc = extract_glibc_version(out)
        if glibc is None or glibc < (2, 27):
            log_warning(f"glibc missing or version too old. Found {glibc}, requirement is >= 2.27")
            return False
    except Exception:
        return False

    try:
        out = subprocess.check_output(["g++", "--version"], text=True)
        version = _parse_gxx_version(out)
        if version is None or version < (9, 0):
            log_warning(f"g++ missing or version too old. Found {version}, requirement is >= 9.0")
            return False
    except Exception:
        return False

    return True


def macos_can_use_prebuilt() -> bool:
    """
    Check if macOS system meets prebuilt binary requirements.
    
    Verifies that the macOS system has Xcode Command Line Tools and
    a compatible clang compiler version for using prebuilt brms bundles.
    
    Returns
    -------
    bool
        True if system meets all requirements:
        - Xcode Command Line Tools installed
        - clang >= 11.0 (for C++17 Stan compilation)
    
    Notes
    -----
    Requirements for prebuilt macOS bundles:
    - Xcode CLT: Provides essential build tools and system headers
    - clang 11+: Minimum compiler for C++17 features used by CmdStan
    
    To install Xcode Command Line Tools:
    ```bash
    xcode-select --install
    ```
    
    Examples
    --------

    ```python
        from brmspy.binaries.env import macos_can_use_prebuilt
        
        if macos_can_use_prebuilt():
            print("System can use prebuilt macOS binaries")
        else:
            print("Install Xcode CLT or upgrade clang")
    ```

    See Also
    --------
    parse_clang_version : Parse clang version from command output
    toolchain_is_compatible : Master toolchain check for all platforms
    """
    try:
        # Check that Xcode CLI tools are installed
        subprocess.check_output(["xcode-select", "-p"], text=True)
    except Exception:
        log_warning(f"xcode cli tools not found")
        return False

    try:
        out = subprocess.check_output(["clang", "--version"], text=True)
        version = parse_clang_version(out)
        if version is None or version < (11, 0):
            log_warning(f"clang missing or version too old. Found {version}, requirement is >= 11.0")
            return False
    except Exception:
        return False

    return True




def windows_can_use_prebuilt() -> bool:
    """
    Check if Windows system meets prebuilt binary requirements.
    
    Verifies that the Windows system has Rtools with MinGW g++ compiler
    required for using prebuilt brms runtime bundles.
    
    Returns
    -------
    bool
        True if system meets all requirements:
        - Rtools toolchain installed (MinGW g++ available)
        - g++ >= 9.0 (for C++17 Stan compilation)
    
    Notes
    -----
    Requirements for prebuilt Windows bundles:
    - Rtools 4.0+: Provides MinGW-w64 toolchain for Stan compilation
    - g++ 9+: Minimum compiler for C++17 features used by CmdStan
    
    Rtools can be downloaded from:
    https://cran.r-project.org/bin/windows/Rtools/
    
    Examples
    --------

    ```python
    from brmspy.binaries.env import windows_can_use_prebuilt
    
    if windows_can_use_prebuilt():
        print("System can use prebuilt Windows binaries")
    else:
        print("Install Rtools 4.0+")
    ```

    See Also
    --------
    parse_gxx_version : Parse g++ version from command output
    toolchain_is_compatible : Master toolchain check for all platforms
    """
    return _windows_has_rtools(silent=True)


# ----- Platform & toolchain gates -----

def supported_platform() -> bool:
    """
    Check if current platform is officially supported for prebuilt binaries.
    
    Validates that the operating system and architecture combination is
    one of the officially supported platforms for prebuilt brms bundles.
    
    Returns
    -------
    bool
        True if platform is in the supported list
    
    Notes
    -----
    Currently supported platforms:
    - linux-x86_64: Linux on x86_64/AMD64 processors
    - macos-x86_64: macOS on Intel processors
    - macos-arm64: macOS on Apple Silicon (M1/M2/M3)
    - windows-x86_64: Windows on x86_64/AMD64 processors
    
    Unsupported combinations (e.g., linux-arm64, windows-arm64) will
    return False even if technically compatible.
    
    Examples
    --------

    ```
    from brmspy.binaries.env import supported_platform
    
    if supported_platform():
        print("Platform is supported for prebuilts")
        # Further checks: toolchain, R version, etc.
    else:
        print("Platform not supported, must build from source")
    ```

    See Also
    --------
    _normalized_os_arch : Get normalized platform identifiers
    can_use_prebuilt : Complete prebuilt eligibility check
    """
    os_name, arch = _normalized_os_arch()
    supported_os = ("linux", "macos", "windows")
    supported_os_arch = {
        "windows": {"x86_64"},
        "linux": {"x86_64"},
        "macos": {"arm64"}
    }

    if os_name not in supported_os:
        log_warning(f"OS '{os_name}' not supported. Requirements: {supported_os}")
        return False
    
    supported_archs = supported_os_arch.get(os_name, 'UNK')
    if arch not in supported_archs:
        log_warning(f"Architecture '{arch}' not supported for OS '{os_name}'. Requirements: {supported_archs}")
        return False

    return True


def toolchain_is_compatible() -> bool:
    """
    Check if system toolchain is compatible with prebuilt binaries.
    
    Routes to the appropriate platform-specific toolchain verification
    function based on the detected operating system.
    
    Returns
    -------
    bool
        True if system has compatible compiler toolchain for current OS
    
    Notes
    -----
    Platform-specific requirements:
    - Linux: glibc >= 2.27, g++ >= 9
    - macOS: Xcode CLT installed, clang >= 11
    - Windows: Rtools installed (MinGW g++ >= 9)
    
    This is a high-level check that delegates to OS-specific functions:
    - `linux_can_use_prebuilt()` for Linux
    - `macos_can_use_prebuilt()` for macOS
    - `windows_can_use_prebuilt()` for Windows
    
    Examples
    --------

    ```
    from brmspy.binaries.env import toolchain_is_compatible
    
    if toolchain_is_compatible():
        print("Toolchain is compatible")
    else:
        print("Upgrade compiler or install build tools")
    ```

    See Also
    --------
    linux_can_use_prebuilt : Linux toolchain check
    macos_can_use_prebuilt : macOS toolchain check
    windows_can_use_prebuilt : Windows toolchain check
    """
    os_name, _ = _normalized_os_arch()
    if os_name == "linux":
        return linux_can_use_prebuilt()
    if os_name == "macos":
        return macos_can_use_prebuilt()
    if os_name == "windows":
        return windows_can_use_prebuilt()
    return False


# ----- Fingerprint + availability -----

def system_fingerprint() -> Optional[str]:
    """
    Generate system fingerprint for prebuilt bundle selection.
    
    Creates a unique identifier string combining OS, architecture, and
    R version to match against available prebuilt runtime bundles.
    
    Returns
    -------
    str or None
        Fingerprint string in format '{os}-{arch}-r{major}.{minor}',
        or None if R version cannot be determined
    
    Notes
    -----
    The fingerprint format is: `{os}-{arch}-r{major}.{minor}`
    
    Example fingerprints:
    - 'linux-x86_64-r4.3'
    - 'macos-arm64-r4.4'
    - 'windows-x86_64-r4.2'
    
    This fingerprint is used to:
    1. Select the correct prebuilt bundle for download
    2. Verify bundle compatibility with current system
    3. Cache bundles in system-specific directories
    
    Examples
    --------

    ```
    from brmspy.binaries.env import system_fingerprint
    
    fp = system_fingerprint()
    if fp:
        print(f"System fingerprint: {fp}")
        # Use fingerprint to select bundle
    else:
        print("Cannot determine R version")
    ```

    See Also
    --------
    _normalized_os_arch : Get OS and architecture
    get_r_version_tuple : Get R version
    prebuilt_available_for : Check bundle availability
    """
    os_name, arch = _normalized_os_arch()
    r_ver = get_r_version_tuple()
    if not r_ver:
        return None
    major, minor, _ = r_ver
    return f"{os_name}-{arch}-r{major}.{minor}"


PREBUILT_FINGERPRINTS: Set[str] = set()


def prebuilt_available_for(fingerprint: Optional[str]) -> bool:
    """
    Check if prebuilt bundle exists for given system fingerprint.
    
    Verifies whether a prebuilt runtime bundle is available for the
    specified system fingerprint. Currently checks against a static set,
    but designed to support dynamic manifest/GitHub Releases lookup.
    
    Parameters
    ----------
    fingerprint : str or None
        System fingerprint string (e.g., 'linux-x86_64-r4.3')
        from `system_fingerprint()`
    
    Returns
    -------
    bool
        True if prebuilt bundle exists for this fingerprint
    
    Notes
    -----
    The `PREBUILT_FINGERPRINTS` set contains all available bundle identifiers.
    This can be populated from:
    - Static manifest file in package
    - Remote JSON manifest from CDN/GitHub
    - GitHub Releases API
    
    Future enhancement: Dynamic manifest fetching to auto-discover new bundles
    without package updates.
    
    Examples
    --------

    ```
        from brmspy.binaries.env import (
            system_fingerprint, prebuilt_available_for
        )
        
        fp = system_fingerprint()
        if fp and prebuilt_available_for(fp):
            print(f"Prebuilt available for {fp}")
        else:
            print("No prebuilt available, must build from source")
    ```

    See Also
    --------
    system_fingerprint : Generate fingerprint for current system
    can_use_prebuilt : Complete eligibility check
    """
    if fingerprint is None:
        return False
    return fingerprint in PREBUILT_FINGERPRINTS


# ----- Top-level gate -----

def can_use_prebuilt() -> bool:
    """
    Master check for prebuilt binary eligibility.
    
    Comprehensive validation that system meets ALL requirements for using
    prebuilt brms runtime bundles. Acts as the primary gate for deciding
    whether to use prebuilts or build from source.
    
    Returns
    -------
    bool
        True only if ALL conditions are met:
        - Platform is supported (OS-architecture combination)
        - R >= 4.2 is available and usable via rpy2
        - System toolchain meets minimum requirements
        - Prebuilt bundle exists for this fingerprint (commented out)
    
    Notes
    -----
    This function performs checks in order of increasing cost:
    1. `supported_platform()`: Quick OS/arch validation
    2. `r_available_and_supported()`: Check R via rpy2
    3. `toolchain_is_compatible()`: Verify compiler versions
    4. `prebuilt_available_for()`: Check bundle availability (currently disabled)
    
    The prebuilt availability check is commented out to allow installation
    even when bundle registry is not yet populated.
    
    Examples
    --------

    ```python
    from brmspy.binaries.env import can_use_prebuilt
    
    if can_use_prebuilt():
        print("Using prebuilt binaries for fast installation")
        # Proceed with prebuilt installation
    else:
        print("Building from source")
        # Proceed with source build
    ```

    ```python
    # Diagnostic check with detailed feedback
    from brmspy.binaries.env import (
        supported_platform, r_available_and_supported,
        toolchain_is_compatible, system_fingerprint,
        prebuilt_available_for
    )
    
    print(f"Platform supported: {supported_platform()}")
    print(f"R available: {r_available_and_supported()}")
    print(f"Toolchain compatible: {toolchain_is_compatible()}")
    fp = system_fingerprint()
    print(f"System fingerprint: {fp}")
    print(f"Bundle available: {prebuilt_available_for(fp)}")
    print(f"Can use prebuilt: {can_use_prebuilt()}")
    ```

    See Also
    --------
    supported_platform : Check platform support
    r_available_and_supported : Check R availability
    toolchain_is_compatible : Check compiler compatibility
    system_fingerprint : Get system identifier
    prebuilt_available_for : Check bundle availability
    """
    if not supported_platform():
        return False

    if not r_available_and_supported():
        return False

    if not toolchain_is_compatible():
        return False

    fp = system_fingerprint()
    #if not prebuilt_available_for(fp):
    #    return False

    return True
