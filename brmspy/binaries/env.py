import platform
import subprocess
from typing import Optional, Tuple, Set, cast


# ----- Helpers: OS / arch -----

def _normalized_os_arch() -> Tuple[str, str]:
    """
    Return normalized (os_name, arch) for fingerprinting and routing.

    os_name: 'linux' | 'macos' | 'windows' | other-raw
    arch: 'x86_64' | 'arm64' | raw
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
    Return (major, minor, patch) for the active R, or None if R/rpy2 not usable.
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


def r_available_and_supported(min_major: int = 4, min_minor: int = 2) -> bool:
    """
    R must be present and >= min_major.min_minor (default: 4.2).
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
    Parse glibc version from `ldd --version` output.
    Returns (major, minor) or None if not found.
    """
    for line in ldd_output.splitlines():
        for token in line.split():
            if token[0].isdigit() and "." in token:
                parts = token.split(".")
                if len(parts) >= 2 and parts[0].isdigit() and parts[1].isdigit():
                    return int(parts[0]), int(parts[1])
    return None


def parse_gxx_version(version_output: str) -> Optional[Tuple[int, int]]:
    """
    Parse g++ major.minor from `g++ --version` output.
    Returns (major, minor) or None on failure.
    """
    for line in version_output.splitlines():
        for token in line.split():
            if token[0].isdigit() and "." in token:
                parts = token.split(".")
                if len(parts) >= 2 and parts[0].isdigit() and parts[1].isdigit():
                    return int(parts[0]), int(parts[1])
    return None


def parse_clang_version(version_output: str) -> Optional[Tuple[int, int]]:
    """
    Parse clang major.minor from `clang --version` output.
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
    Linux prebuilt eligibility:
    - glibc >= 2.27
    - g++ >= 9
    """
    try:
        out = subprocess.check_output(["ldd", "--version"], text=True)
        glibc = extract_glibc_version(out)
        if glibc is None or glibc < (2, 27):
            return False
    except Exception:
        return False

    try:
        out = subprocess.check_output(["g++", "--version"], text=True)
        version = parse_gxx_version(out)
        if version is None or version < (9, 0):
            return False
    except Exception:
        return False

    return True


def macos_can_use_prebuilt() -> bool:
    """
    macOS prebuilt eligibility:
    - Xcode command line tools present (xcode-select -p succeeds)
    - clang >= 11
    """
    try:
        # Check that Xcode CLI tools are installed
        subprocess.check_output(["xcode-select", "-p"], text=True)
    except Exception:
        return False

    try:
        out = subprocess.check_output(["clang", "--version"], text=True)
        version = parse_clang_version(out)
        if version is None or version < (11, 0):
            return False
    except Exception:
        return False

    return True


def windows_can_use_prebuilt() -> bool:
    """
    Windows prebuilt eligibility:
    - RTools toolchain present (mingw g++)
    - g++ reports a reasonable version
    """
    try:
        out = subprocess.check_output(["g++", "--version"], text=True, shell=True)
    except Exception:
        return False

    # Very rough: we expect mingw in the banner
    if "mingw" not in out.lower():
        return False

    version = parse_gxx_version(out)
    if version is None or version < (9, 0):
        return False

    return True


# ----- Platform & toolchain gates -----

def supported_platform() -> bool:
    """
    Only allow prebuilts on the platforms we explicitly support.
    Currently:
      - linux-x86_64
      - macos-x86_64
      - macos-arm64
      - windows-x86_64
    """
    os_name, arch = _normalized_os_arch()

    if os_name not in ("linux", "macos", "windows"):
        return False

    if os_name == "windows" and arch != "x86_64":
        return False

    if os_name == "linux" and arch != "x86_64":
        # You can relax this later if you build Linux ARM prebuilts.
        return False

    if os_name == "macos" and arch not in ("x86_64", "arm64"):
        return False

    return True


def toolchain_is_compatible() -> bool:
    """
    Route to the per-OS toolchain check.
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
    Build a fingerprint for selecting a prebuilt bundle.

    Example: 'linux-x86_64-r4.3'
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
    Return True if we have a prebuilt for this fingerprint.

    For now this just checks against a static set; later you can swap this out
    to consult a JSON manifest or probe GitHub Releases.
    """
    if fingerprint is None:
        return False
    return fingerprint in PREBUILT_FINGERPRINTS


# ----- Top-level gate -----

def can_use_prebuilt() -> bool:
    """
    Hard gate: only return True if we're on a supported platform,
    with a supported R, a compatible toolchain, and a known prebuilt bundle.
    """
    if not supported_platform():
        return False

    if not r_available_and_supported():
        return False

    if not toolchain_is_compatible():
        return False

    fp = system_fingerprint()
    if not prebuilt_available_for(fp):
        return False

    return True
