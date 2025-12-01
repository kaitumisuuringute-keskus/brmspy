import os
import platform
import subprocess
import tempfile
from typing import Optional, Tuple, cast
from urllib import request
from packaging.version import Version
from rpy2 import robjects as ro

from brmspy.helpers.log import log, log_warning

def _get_r_version() -> Version:
    """
    Get current R version from active R installation via rpy2.
    
    Queries the embedded R session to retrieve version information and
    converts it to a packaging.Version object for version comparisons.
    
    Returns
    -------
    packaging.version.Version
        R version as Version object (e.g., Version("4.3.2"))
    
    Notes
    -----
    This function uses rpy2 to call R's `getRversion()` function and
    converts the result to Python's packaging.Version for semantic
    version comparisons.
    
    Examples
    --------

    ```python
    from brmspy.helpers.rtools import _get_r_version
    
    r_ver = _get_r_version()
    print(r_ver)  # e.g., "4.3.2"
    
    # Version comparisons
    if r_ver >= Version("4.3.0"):
        print("R 4.3+")
    ```

    See Also
    --------
    pick_rtools_for_r : Select appropriate Rtools version for R
    """
    ver_str = str(cast(list, ro.r('as.character(getRversion())'))[0])
    return Version(ver_str)


def pick_rtools_for_r(r_ver: Version) -> str | None:
    """
    Select appropriate Rtools version tag for given R version.
    
    Maps R version to compatible Rtools major version. Rtools is required
    on Windows to compile Stan models and R packages with C++ code.
    
    Parameters
    ----------
    r_ver : packaging.version.Version
        R version to match
    
    Returns
    -------
    str or None
        Rtools version tag ('40', '42', '43', '44', '45') or None if
        R version is too old (< 4.0.0) for automatic handling
    
    Notes
    -----
    **R to Rtools Version Mapping:**
    
    - R 4.0.x - 4.1.x: Rtools 40
    - R 4.2.x: Rtools 42
    - R 4.3.x: Rtools 43
    - R 4.4.x: Rtools 44
    - R 4.5.x+: Rtools 45
    - R < 4.0.0: None (legacy, not supported)
    
    **Rtools Purpose:**
    
    Rtools provides MinGW-w64 compiler toolchain on Windows for:
    - Compiling Stan models (required by cmdstanr)
    - Building R packages from source
    - C++ compilation for brms/Stan
    
    Examples
    --------

    ```python
    from packaging.version import Version
    from brmspy.helpers.rtools import pick_rtools_for_r
    
    # R 4.3.2 needs Rtools 43
    tag = pick_rtools_for_r(Version("4.3.2"))
    print(tag)  # "43"
    
    # R 4.4.1 needs Rtools 44
    tag = pick_rtools_for_r(Version("4.4.1"))
    print(tag)  # "44"
    
    # Legacy R not supported
    tag = pick_rtools_for_r(Version("3.6.3"))
    print(tag)  # None
    ```

    See Also
    --------
    _install_rtools_for_current_r : Automatically install matching Rtools
    _get_r_version : Get current R version
    
    References
    ----------
    .. [1] Rtools download page: https://cran.r-project.org/bin/windows/Rtools/
    """
    if r_ver < Version("4.0.0"):
        return None  # legacy, not worth supporting in brmspy
    if r_ver < Version("4.2.0"):
        return "40"  # R 4.0.x–4.1.x
    if r_ver < Version("4.3.0"):
        return "42"  # R 4.2.x
    if r_ver < Version("4.4.0"):
        return "43"  # R 4.3.x
    if r_ver < Version("4.5.0"):
        return "44"  # R 4.4.x
    if r_ver < Version("4.6.0"):
        return "45"
    if r_ver < Version("4.7.0"):
        return "47"

    return "46"      # R 4.5.x and up


RTOOLS_INSTALLERS = {
    "40": "https://cran.r-project.org/bin/windows/Rtools/rtools40-x86_64.exe",
    "42": "https://cran.r-project.org/bin/windows/Rtools/rtools42/files/rtools42-5355-5357.exe",
    "43": "https://cran.r-project.org/bin/windows/Rtools/rtools43/files/rtools43-5976-5975.exe",
    "44": "https://cran.r-project.org/bin/windows/Rtools/rtools44/files/rtools44-6459-6401.exe",
    "45": "https://cran.r-project.org/bin/windows/Rtools/rtools45/files/rtools45-6691-6492.exe",
}

def _silent_install_exe(url: str, label: str) -> None:
    """
    Download and silently install Windows executable.
    
    Downloads installer from URL to temp directory and runs it with
    silent installation flags. Used for automated Rtools installation.
    
    Parameters
    ----------
    url : str
        Direct download URL for .exe installer
    label : str
        Label for downloaded file (e.g., "rtools44")
    
    Raises
    ------
    subprocess.CalledProcessError
        If installer execution fails
    
    Notes
    -----
    **Silent Installation Flags:**
    
    Uses Inno Setup silent install flags:
    - `/VERYSILENT`: No GUI, no progress window
    - `/SUPPRESSMSGBOXES`: Suppress message boxes
    - `/NORESTART`: Don't restart system after install
    
    **Download Location:**
    
    Installer is downloaded to system temp directory (tempfile.gettempdir()).
    
    **Subprocess Behavior:**
    
    Blocks until installation completes. Installation typically takes 1-2 minutes.
    
    Examples
    --------

    ```python
    from brmspy.helpers.rtools import _silent_install_exe
    
    # Install Rtools 44 silently
    url = "https://cran.r-project.org/bin/windows/Rtools/rtools44/files/rtools44-6459-6401.exe"
    _silent_install_exe(url, "rtools44")
    # Rtools 44 now installed to C:\\rtools44
    ```

    See Also
    --------
    _install_rtools_for_current_r : High-level Rtools installation
    """
    tmp_dir = tempfile.gettempdir()
    exe_path = os.path.join(tmp_dir, f"{label}.exe")
    log(f"downloading {label} from {url}")
    request.urlretrieve(url, exe_path)
    log(f"running {label} installer silently...")
    subprocess.run(
        [
            exe_path,
            "/VERYSILENT",
            "/SUPPRESSMSGBOXES",
            "/NORESTART",
        ],
        check=True,
    )


def _parse_gxx_version(version_output: str) -> Optional[Tuple[int, int]]:
    """
    Parse g++ compiler version from command output.
    
    Extracts g++ version number from the output of `g++ --version`.
    Used to verify minimum compiler requirement for building Stan models.
    
    Parameters
    ----------
    version_output : str
        Output from `g++ --version` command
    
    Returns
    -------
    tuple of (int, int) or None
        (major, minor) version, or None if parsing fails
    
    Examples
    --------

    ```python
    import subprocess
    
    out = subprocess.check_output(["g++", "--version"], text=True)
    version = parse_gxx_version(out)
    if version and version >= (9, 0):
        print("g++ 9+ available")
    ```

    See Also
    --------
    linux_can_use_prebuilt : Linux compatibility check
    windows_can_use_prebuilt : Windows (MinGW) compatibility check
    """
    for line in version_output.splitlines():
        for token in line.split():
            if token[0].isdigit() and "." in token:
                parts = token.split(".")
                if len(parts) >= 2 and parts[0].isdigit() and parts[1].isdigit():
                    return int(parts[0]), int(parts[1])
    return None

def _windows_has_rtools(silent=False) -> bool:
    # If Rtools is already found, skip (cmdstanr check)
    try:
        # Check if 'make' is available. If yes, we probably have Rtools.
        make_path = str(cast(list, ro.r('Sys.which("make")'))[0])
        if make_path and "rtools" in make_path.lower():
            return True
    except Exception:
        pass

    try:
        out = subprocess.check_output(["g++", "--version"], text=True, shell=True)
    except Exception:
        if not silent:
            log_warning(f"g++ not found")
        return False

    # Very rough: we expect mingw in the banner
    if "mingw" not in out.lower():
        if not silent:
            log_warning(f"mingw not found in g++ banner")
        return False

    version = _parse_gxx_version(out)
    if version is None or version < (9, 0):
        if not silent:
            log_warning(f"g++ version too old. Found {version}, requirement is >= 9.0")
        return False
    return True

def _install_rtools_for_current_r() -> Optional[str]:
    """
    Auto-detect R version and install matching Rtools if needed (Windows only).
    
    Detects current R version via rpy2, determines appropriate Rtools version,
    checks if already installed, and installs if missing. Updates PATH to
    include Rtools binaries for both Python and R sessions.
    
    Parameters
    ----------
    
    Returns
    -------
    str or None
        Rtools version tag installed ('40', '42', '43', '44', '45'),
        or None if:
        - Not on Windows
        - R version too old/new for automatic handling
        - Rtools already present
        - CI safety check failed
    
    Notes
    -----
    
    **Installation Process:**
    
    1. Check if Windows (return None otherwise)
    2. Check CI safety flag
    3. Detect R version via rpy2
    4. Map R version to Rtools version
    5. Check if make command available (indicates Rtools present)
    6. Download and silently install Rtools if missing
    7. Update PATH for Python subprocesses
    8. Update R's PATH environment directly via rpy2
    9. Force R to re-scan for build tools (pkgbuild)
    
    **PATH Updates:**
    
    After installation, adds these directories to PATH:
    - `C:\\rtools{tag}\\usr\\bin`: Unix utilities
    - `C:\\rtools{tag}\\mingw64\\bin`: MinGW compiler
    
    **R Environment Update:**
    
    Critical: Since rpy2 has already started R, PATH changes must be
    applied directly to R's environment via `Sys.setenv()` in addition
    to Python's `os.environ`.
    
    Examples
    --------

    ```python
    from brmspy.helpers.rtools import _install_rtools_for_current_r
    
    tag = _install_rtools_for_current_r()
    if tag:
        print(f"Rtools {tag} installed/verified")
    ```

    See Also
    --------
    pick_rtools_for_r : Determine Rtools version for R version
    _silent_install_exe : Download and install executable
    _get_r_version : Get current R version
    
    References
    ----------
    .. [1] Rtools information: https://cran.r-project.org/bin/windows/Rtools/
    .. [2] cmdstanr Rtools setup: https://mc-stan.org/cmdstanr/articles/cmdstanr.html#installation
    """
    if platform.system() != "Windows":
        return None


    r_ver = _get_r_version()
    tag = pick_rtools_for_r(r_ver)
    
    if _windows_has_rtools():
        return tag

    if tag is None:
        log_warning(f"R {r_ver} is too old/new for automatic Rtools handling.")
        return None

    url = RTOOLS_INSTALLERS.get(tag)
    if not url:
        log_warning(f"no installer URL configured for Rtools{tag}")
        return None

    _silent_install_exe(url, f"rtools{tag}")

    # Best-effort PATH tweaks
    root = rf"C:\rtools{tag}"
    candidates = [
        os.path.join(root, "usr", "bin"),
        os.path.join(root, "x86_64-w64-mingw32.static.posix", "bin"),
        os.path.join(root, "mingw64", "bin"),
    ]
    
    new_paths = []
    for p in candidates:
        if os.path.isdir(p):
            new_paths.append(p)
            log(f"found Rtools bin: {p}")

    if new_paths:
        # Update Python environment (for subprocesses)
        path_sep = os.pathsep
        full_new_path = path_sep.join(new_paths)
        os.environ["PATH"] = full_new_path + path_sep + os.environ.get("PATH", "")

        # CRITICAL: Update R's environment directly!
        # rpy2 has already started, so R won't see the os.environ change unless we force it.
        r_path_cmd = f'Sys.setenv(PATH = paste("{full_new_path}", Sys.getenv("PATH"), sep="{path_sep}"))'
        ro.r(r_path_cmd)
        
        # 3. Force R to re-scan for build tools
        ro.r("""
        if (requireNamespace("pkgbuild", quietly = TRUE)) {
            pkgbuild::find_rtools(debug = TRUE)
        }
        """)

    return tag