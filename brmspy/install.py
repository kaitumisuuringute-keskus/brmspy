import os
import shutil
import subprocess
from typing import Optional, Union, cast, Tuple
from packaging.version import Version
import multiprocessing
import platform

import rpy2.robjects as ro
from rpy2.robjects.packages import importr
from rpy2.robjects.vectors import StrVector

from brmspy.binaries.use import install_and_activate_runtime
from brmspy.helpers.rtools import _install_rtools_for_current_r
from brmspy.helpers.singleton import _get_brms, _invalidate_singletons
from build.lib.brmspy.helpers.rtools import _get_r_version

def _parse_version_spec(spec: Optional[str]) -> Tuple[str, Optional[Version]]:
    """
    Parse version specification string into operator and Version object.
    
    Supports version constraints commonly used in package managers:
    bare version ("2.21.0") treated as exact match, explicit operators
    ("==2.21.0", ">=2.20.0", "<=2.23.0"), or None/empty for any version.
    
    Parameters
    ----------
    spec : str or None
        Version specification string. Formats:
        - None or empty: accept any version
        - "2.21.0": exact version (treated as ==)
        - "==2.21.0": exact version
        - ">=2.20.0": minimum version
        - "<=2.23.0": maximum version
    
    Returns
    -------
    tuple of (str, Version or None)
        (operator, version) where operator is one of:
        - "any": no version constraint
        - "==": exact version match
        - ">=": minimum version
        - "<=": maximum version
    
    Examples
    --------
    ```python
    _parse_version_spec("2.21.0")
    # ('==', <Version('2.21.0')>)
    
    _parse_version_spec(">=2.20.0")
    # ('>=', <Version('2.20.0')>)
    
    _parse_version_spec(None)
    # ('any', None)
    ```
    """
    if spec is None:
        return "any", None
    spec = spec.strip()
    for op in ("<=", ">=", "=="):
        if spec.startswith(op):
            return op, Version(spec[len(op):].strip())
    # bare version => '=='
    return "==", Version(spec)

def _satisfies(installed: Version, op: str, required: Optional[Version]) -> bool:
    """
    Check if installed version satisfies version constraint.
    
    Evaluates version requirements using packaging.Version comparison.
    Used to determine if package upgrade is needed.
    
    Parameters
    ----------
    installed : Version
        Currently installed package version
    op : str
        Comparison operator: "any", "==", ">=", or "<="
    required : Version or None
        Required version, or None if op is "any"
    
    Returns
    -------
    bool
        True if installed version satisfies constraint
    
    Examples
    --------
    ```python
    from packaging.version import Version
    
    _satisfies(Version("2.21.0"), ">=", Version("2.20.0"))
    # True
    
    _satisfies(Version("2.19.0"), ">=", Version("2.20.0"))
    # False
    
    _satisfies(Version("2.21.0"), "any", None)
    # True
    ```
    """
    if op == "any" or required is None:
        return True
    if op == "==":
        return installed == required
    if op == ">=":
        return installed >= required
    if op == "<=":
        return installed <= required
    return True

def _get_r_pkg_version(package: str) -> Optional[Version]:
    """
    Get installed R package version.
    
    Queries R's package system via rpy2 to retrieve the installed
    version of a package. Returns None if package is not installed.
    
    Parameters
    ----------
    package : str
        R package name (e.g., "brms", "cmdstanr", "rstan")
    
    Returns
    -------
    Version or None
        Package version if installed, None otherwise
    
    Examples
    --------
    ```python
    _get_r_pkg_version("brms")
    # <Version('2.21.0')>
    
    _get_r_pkg_version("nonexistent_package")
    # None
    ```
    """
    try:
        # utils::packageVersion("pkg") -> "x.y.z"
        v_str = cast(list, ro.r(f"as.character(utils::packageVersion('{package}'))"))[0]
        return Version(v_str)
    except Exception:
        return None
    
def _install_exact_version(package: str, version: Version, repos) -> None:
    """
    Install specific R package version using remotes::install_version().
    
    Pins installation to exact version, useful for reproducibility.
    Requires the remotes R package, which is auto-installed if missing.
    
    Parameters
    ----------
    package : str
        R package name
    version : Version
        Exact version to install
    repos : list of str
        Repository URLs for package installation
    
    Raises
    ------
    Exception
        If installation fails or package version unavailable
    
    Notes
    -----
    Uses remotes::install_version() which downloads packages from CRAN
    archives. Historical versions may not always be available, especially
    for very recent or very old packages.
    
    Examples
    --------
    ```python
    from packaging.version import Version
    
    _install_exact_version(
        "brms",
        Version("2.21.0"),
        ["https://cloud.r-project.org"]
    )
    ```
    """
    # ensure remotes is available
    ro.r('if (!requireNamespace("remotes", quietly = TRUE)) '
         'install.packages("remotes", repos="https://cloud.r-project.org")')

    # pass repos vector from Python into R
    ro.globalenv[".brmspy_repos"] = StrVector(repos)
    ro.r(f'remotes::install_version("{package}", '
         f'version="{version}", repos=.brmspy_repos)')
    del ro.globalenv[".brmspy_repos"]


def _get_linux_repo():
    """
    Detect Linux distribution and return Posit Package Manager URL.
    
    Reads /etc/os-release to identify the Linux distribution codename
    (e.g., "jammy", "focal") and constructs the appropriate Posit
    Package Manager binary repository URL. Falls back to Ubuntu 22.04
    (jammy) if detection fails.
    
    Returns
    -------
    str
        Posit Package Manager repository URL for detected distribution
    
    Notes
    -----
    Posit Package Manager (P3M) provides precompiled R packages for Linux,
    dramatically speeding up installation by avoiding source compilation.
    The URL format includes the Linux distribution codename to ensure
    binary compatibility with the system's glibc version.
    
    Examples
    --------

    ```python
        _get_linux_repo()  # On Ubuntu 22.04
        # 'https://packagemanager.posit.co/cran/__linux__/jammy/latest'
        
        _get_linux_repo()  # On Ubuntu 20.04
        # 'https://packagemanager.posit.co/cran/__linux__/focal/latest'
    ```
    """
    try:
        with open("/etc/os-release") as f:
            lines = f.readlines()
            
        codename = "jammy" # Default fallback (Ubuntu 22.04)
        for line in lines:
            if line.startswith("VERSION_CODENAME="):
                codename = line.strip().split("=")[1].strip('"')
                break
                
        return f"https://packagemanager.posit.co/cran/__linux__/{codename}/latest"
    except FileNotFoundError:
        return "https://packagemanager.posit.co/cran/__linux__/jammy/latest"


def _install_rpackage(
    package: str,
    version: Optional[str] = None,
    repos_extra: Optional[Union[str, list[Optional[str]]]] = None,
):
    """
    Ensure an R package is installed, with optional version constraint.

    Parameters
    ----------
    package : str
        R package name.
    version : str | None
        Version spec. Supported forms:
          - None        -> any version is fine
          - '2.21.0'    -> exact, equivalent to '==2.21.0'
          - '==2.21.0'  -> exact version via remotes::install_version
          - '>=2.21.0'  -> require at least this version
          - '<=2.20.0'  -> require at most this version
        For >= / <= we *check* the constraint; if it fails we install/upgrade
        using the normal CRAN path (latest from the configured repos).
    repos_extra : str | list[str] | None
        Extra repositories to append in addition to CRAN / binary repo.
    """
    if version == "latest":
        version = None
    utils = importr("utils")
    system = platform.system()
    cores = multiprocessing.cpu_count()

    op, required_version = _parse_version_spec(version)

    repos: list[str] = ["https://cloud.r-project.org"]  # good default mirror

    if system == "Linux":
        # On Linux, we MUST use P3M to get binaries.
        binary_repo = _get_linux_repo()
        repos.insert(0, binary_repo)  # high priority
        preferred_type = "source"     # P3M binaries look like source to R
    else:
        # Windows / macOS use native CRAN binaries
        preferred_type = "binary"

    if repos_extra:
        if isinstance(repos_extra, list):
            for _r in repos_extra:
                if isinstance(_r, str) and _r not in repos:
                    repos.append(_r)
        elif repos_extra not in repos:
            repos.append(repos_extra)

    # --- check current installation + version ---
    try:
        importr(package)
        installed_version = _get_r_pkg_version(package)
    except Exception:
        installed_version = None

    if installed_version is not None:
        if _satisfies(installed_version, op, required_version):
            print(
                f"brmspy: {package} {installed_version} already installed "
                f"and satisfies '{op} {required_version}'." if required_version else
                f"brmspy: {package} {installed_version} already installed."
            )
            return
        else:
            print(
                f"brmspy: {package} {installed_version} does NOT satisfy "
                f"'{op} {required_version}', reinstalling..."
            )

    print(f"brmspy: Installing {package} on {system} (Repos: {len(repos)})...")

    # --- exact pin via remotes::install_version ---
    if op == "==" and required_version is not None:
        try:
            _install_exact_version(package, required_version, repos)
            print(f"brmspy: Installed {package}=={required_version} via remotes::install_version.")
            return
        except Exception as e:
            print(f"Failed to install exact {package}=={required_version}: {e}")
            raise

    # --- normal path: use install.packages with preferred type, then fallback to source ---
    try:
        # Primary Attempt (Fast Binary / P3M)
        utils.install_packages(
            StrVector((package,)),
            repos=StrVector(repos),
            type=preferred_type,
            Ncpus=cores,
        )
        installed_version = _get_r_pkg_version(package)
        if installed_version is None:
            raise RuntimeError(f"{package} did not appear after install (type={preferred_type}).")
        print(f"brmspy: Installed {package} via {preferred_type} path.")
    except Exception as e:
        print(f"{preferred_type} install failed. Falling back to source compilation. ({e})")
        try:
            utils.install_packages(
                StrVector((package,)),
                repos=StrVector(repos),
                # dont set type, let R manage this.
                Ncpus=cores,
            )
            installed_version = _get_r_pkg_version(package)
            if installed_version is None:
                raise RuntimeError(f"{package} did not appear after source install.")
            print(f"brmspy: Installed {package} from source.")
        except Exception as e2:
            print(f"Failed to install {package}.")
            raise e2



def _build_cmstanr():
    """
    Build and configure CmdStan compiler via cmdstanr.
    
    Downloads and compiles the CmdStan probabilistic programming language
    compiler, which brms uses for model fitting. Handles Windows toolchain
    verification and automatic Rtools installation if needed.
    
    The build process:
    1. Verifies Windows toolchain (Rtools) if on Windows
    2. Auto-installs Rtools if missing (CI only)
    3. Downloads CmdStan source from GitHub
    4. Compiles CmdStan using available CPU cores
    5. Configures cmdstanr to use the compiled installation
    
    Raises
    ------
    Exception
        If toolchain verification fails on Windows
        If CmdStan compilation fails
    
    Notes
    -----
    **Performance**: Uses (n_cores - 1) for compilation to leave one core
    available for system responsiveness. On systems with <= 4 cores, uses
    all cores.
    
    **Windows Requirements**: Requires Rtools with compatible version:
    - R 4.0-4.1: Rtools40
    - R 4.2: Rtools42
    - R 4.3: Rtools43
    - R 4.4: Rtools44
    - R 4.5+: Rtools45
    
    **Linux/macOS**: Requires C++ compiler (g++ >= 9 or clang >= 11)
    
    See Also
    --------
    install_brms : Main installation function that calls this
    cmdstanr::install_cmdstan : R documentation
        https://mc-stan.org/cmdstanr/reference/install_cmdstan.html
    
    Examples
    --------
    ```python
    from brmspy.install import _build_cmstanr
    _build_cmstanr()  # Downloads and compiles CmdStan
    ```
    """
    cores = multiprocessing.cpu_count()
    if cores > 4:
        cores -= 1

    ro.r("library(cmdstanr)")

    if platform.system() == "Windows":
        print("brmspy: Checking Windows toolchain (Rtools/cmdstanr)...")
        try:
            ro.r("cmdstanr::check_cmdstan_toolchain(fix = TRUE)")
        except Exception as e:
            print(f"Toolchain check failed: {e}")
            tag = _install_rtools_for_current_r()
            if not tag:
                print(
                    "brmspy: Rtools auto-install failed or disabled. "
                    "Please install the matching Rtools version manually: "
                    "https://cran.r-project.org/bin/windows/Rtools/"
                )
                return
            if tag != "45":
                # cmdstanr does an invalid mapping of r4.5 -> 44
                # only check if tag isnt 45
                print(f"brmspy: Installed Rtools{tag}, re-checking toolchain...")
                ro.r("cmdstanr::check_cmdstan_toolchain(fix = TRUE)")

    ro.r(f"cmdstanr::install_cmdstan(cores = {cores}, overwrite = FALSE)")

def install_prebuilt(runtime_version="0.1.0", url: Optional[str] = None, bundle: Optional[str] = None):
    """
    Install prebuilt brmspy runtime bundle for fast setup.
    
    Downloads and activates a precompiled runtime containing:
    - R packages (brms, cmdstanr, dependencies)
    - Compiled CmdStan binary
    - Complete environment ready for immediate use
    
    This reduces setup time from ~30 minutes to ~1 minute by avoiding
    compilation. Available for specific platform/R version combinations.
    
    Parameters
    ----------
    runtime_version : str, default="0.1.0"
        Runtime schema version (not pip version)
    url : str, optional
        Custom URL for runtime bundle. If None, uses GitHub releases
    bundle : str, optional
        Local path to runtime bundle (.tar.gz or directory)
    
    Returns
    -------
    bool
        True if installation succeeded, False otherwise
    
    Raises
    ------
    RuntimeError
        If prebuilt binaries not available for this platform
    
    Notes
    -----
    **Platform Support**: Prebuilt binaries are available for:
    - Linux: x86_64, glibc >= 2.27, g++ >= 9
    - macOS: x86_64 and arm64, clang >= 11
    - Windows: x86_64 with Rtools
    
    **R Version**: Runtime includes all R packages, so they must match
    your R installation's major.minor version (e.g., R 4.3.x).
    
    **System Fingerprint**: Runtime is selected based on:
    - Operating system (linux/macos/windows)
    - CPU architecture (x86_64/arm64)
    - R version (major.minor)
    
    Example: `linux-x86_64-r4.3`
    
    See Also
    --------
    install_brms : Main installation function
    brmspy.binaries.install_and_activate_runtime : Low-level installer
    brmspy.binaries.system_fingerprint : Platform detection
    
    Examples
    --------
    Install from GitHub releases:

    ```python
    from brmspy.install import install_prebuilt
    install_prebuilt()
    ```

    Install from local bundle:
    
    ```python
    install_prebuilt(bundle="/path/to/runtime.tar.gz")
    ```

    Install from custom URL:
    
    ```python
        install_prebuilt(url="https://example.com/runtime.tar.gz")
    ```
    """
    from brmspy.binaries import env
    if not env.can_use_prebuilt():
        raise RuntimeError(
            "Prebuilt binaries are not available for your system. "
            "Please install brms manually or in install_brms set use_prebuilt_binaries=False."
        )
    
    if platform.system() == "Windows":
        rtools_tag = _install_rtools_for_current_r()

    fingerprint = env.system_fingerprint()
    if url is None and bundle is None:
        url = f"https://github.com/kaitumisuuringute-keskus/brmspy/releases/download/runtime/brmspy-runtime-{runtime_version}-{fingerprint}.tar.gz"

    try:
        return install_and_activate_runtime(
            url=url,
            bundle=bundle,
            runtime_version=runtime_version
        )
    except Exception as e:
        print(f"{e}")
        return False

def install_brms(
    brms_version: str = "latest",
    repo: Optional[str] = None,
    install_cmdstanr: bool = True,
    install_rstan: bool = False,
    cmdstanr_version: str = "latest",
    rstan_version: str = "latest",
    use_prebuilt_binaries = False
):
    """
    Install brms R package, optionally cmdstanr and CmdStan compiler, or rstan.
    
    Parameters
    ----------
    brms_version : str, default="latest"
        brms version: "latest", "2.23.0", or ">=2.20.0"
    repo : str | None, default=None
        Extra CRAN repository URL
    install_cmdstanr : bool, default=True
        Whether to install cmdstanr and build CmdStan compiler
    install_rstan : bool, default=False
        Whether to install rstan (alternative to cmdstanr)
    cmdstanr_version : str, default="latest"
        cmdstanr version: "latest", "0.8.1", or ">=0.8.0"
    rstan_version : str, default="latest"
        rstan version: "latest", "2.32.6", or ">=2.32.0"
    use_prebuilt_binaries: bool, default=False
        Uses fully prebuilt binaries for cmdstanr and brms and their dependencies. 
        Ignores system R libraries and uses the latest brms and cmdstanr available 
        for your system. Requires R>=4 and might not be compatible with some older
        systems or missing toolchains. Can reduce setup time by 50x.
    
    Examples
    --------
    Basic installation:
    
    ```python
    from brmspy import brms
    brms.install_brms()
    ```
    Install specific version:
    
    ```python
    brms.install_brms(brms_version="2.23.0")
    ```

    Use rstan instead of cmdstanr:

    ```python
    brms.install_brms(install_cmdstanr=False, install_rstan=True)
    ```

    Fast installation with prebuilt binaries:
    ```python
    brms.install_brms(use_prebuilt_binaries=True)
    ```
    """
    if use_prebuilt_binaries:
        if install_prebuilt():
            print("\nSetup complete! You're ready to use brmspy.")
            return

    print("Installing brms...")
    _install_rpackage("brms", version=brms_version, repos_extra=[repo])

    if install_cmdstanr:
        if platform.system() == "Windows" and _get_r_version() >= Version("4.5.0") and not cmdstanr_version:
            # cmdstanr <0.9 does not recognise rtools 45.
            cmdstanr_version = ">=0.9.0"

        print("Installing cmdstanr...")
        _install_rpackage("cmdstanr", version=cmdstanr_version, repos_extra=["https://mc-stan.org/r-packages/", repo])
        print("Building cmdstanr...")
        _build_cmstanr()

    if install_rstan:
        print("Installing rstan...")
        _install_rpackage("rstan", version=rstan_version, repos_extra=[repo])

    _invalidate_singletons()
    # Import to mitigate lazy imports
    _get_brms()

    print("\nSetup complete! You're ready to use brmspy.")