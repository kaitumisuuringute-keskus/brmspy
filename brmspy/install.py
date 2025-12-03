import os
import shutil
import subprocess
from typing import Callable, List, Optional, Union, cast, Tuple
from packaging.version import Version
import multiprocessing
import platform

import rpy2.robjects as ro
from rpy2.robjects.packages import importr
from rpy2.robjects.vectors import StrVector

from brmspy.binaries.use import install_and_activate_runtime
from brmspy.binaries.r import _forward_github_token_to_r, _get_r_pkg_installed, _get_r_pkg_version, _try_force_unload_package
from brmspy.helpers.log import greet, log, LogTime
from brmspy.helpers.rtools import _install_rtools_for_current_r
from brmspy.helpers.singleton import _get_brms, _invalidate_singletons
from brmspy.helpers.rtools import _get_r_version
from brmspy.helpers.log import log_error, log_warning

def _init():
    # Set the CRAN mirror globally for this session. 
    # This prevents 'build-manifest.R' or subsequent installs from prompting for a mirror.
    ro.r('options(repos = c(CRAN = "https://cloud.r-project.org"))')


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
        Version spec. Semantics when not None are delegated entirely to
        remotes::install_version(), so you get full support for:

          - Exact version:
              "2.21.0"
          - Single constraint:
              ">= 2.21.0"
              "< 2.23.0"
          - Multiple constraints:
              ">= 1.12.0, < 1.14"
              c(">= 1.12.0", "< 1.14")  # if you pass a vector via R

        Special cases handled here:
          - None / "" / "latest" / "any"  -> no constraint, use install.packages()
    repos_extra : str | list[str] | None
        Extra repositories to append in addition to CRAN / binary repo.
    """
    # Normalise special values that mean "latest / no constraint"
    if version is not None:
        v = version.strip()
        if v == "" or v.lower() in ("latest", "any"):
            version = None
        else:
            version = v

    utils = importr("utils")
    system = platform.system()
    cores = multiprocessing.cpu_count()

    already_installed = _get_r_pkg_installed(package)

    repos: list[str] = ["https://cloud.r-project.org"]  # good default mirror

    if system == "Linux":
        # On Linux, we MUST use P3M to get binaries. These present as "source"
        # to R, so type="source" is actually fine.
        binary_repo = _get_linux_repo()
        repos.insert(0, binary_repo)  # high priority
        preferred_type = "source"
    else:
        # Windows / macOS use native CRAN binaries in the "no version" path
        preferred_type = "binary"

    if repos_extra:
        if isinstance(repos_extra, list):
            for _r in repos_extra:
                if isinstance(_r, str) and _r not in repos:
                    repos.append(_r)
        elif repos_extra not in repos:
            repos.append(repos_extra)

    # ------------------------------------------------------------------
    # BRANCH 1: version *specified* -> delegate entirely to remotes
    # ------------------------------------------------------------------
    if version is not None:
        log(
            f"Installing {package} "
            f"(version spec: {version!r}) via remotes::install_version()..."
        )

        # Ensure remotes is available
        ro.r(
            'if (!requireNamespace("remotes", quietly = TRUE)) '
            'install.packages("remotes", repos = "https://cloud.r-project.org")'
        )

        # Pass repo vector from Python into R
        ro.globalenv[".brmspy_repos"] = StrVector(repos)

        # Escape double quotes in version spec just in case
        v_escaped = version.replace('"', '\\"')

        try:
            if already_installed and system == "Windows":
                _try_force_unload_package(package)
            ro.r(
                f'remotes::install_version('
                f'package = "{package}", '
                f'version = "{v_escaped}", '
                f'repos = .brmspy_repos)'
            )
        finally:
            # Clean up
            del ro.globalenv[".brmspy_repos"]

        installed_version = _get_r_pkg_version(package)
        if installed_version is None:
            raise RuntimeError(
                f"{package} did not appear after remotes::install_version('{version}')."
            )

        log(
            f"Installed {package} via remotes::install_version "
            f"(installed: {installed_version})."
        )
        return

    # ------------------------------------------------------------------
    # BRANCH 2: no version spec -> "latest" from repos via install.packages
    # ------------------------------------------------------------------
    installed_version = None
    try:
        if already_installed:
            installed_version = _get_r_pkg_version(package)
    except Exception:
        installed_version = None

    if installed_version is not None:
        log(f"{package} {installed_version} already installed.")
        return

    log(f"Installing {package} on {system} (Repos: {len(repos)})...")

    try:
        # Primary Attempt (Fast Binary / P3M)
        if already_installed and system == "Windows":
            _try_force_unload_package(package)
        utils.install_packages(
            StrVector((package,)),
            repos=StrVector(repos),
            type=preferred_type,
            Ncpus=cores,
        )
        installed_version = _get_r_pkg_version(package)
        if installed_version is None:
            raise RuntimeError(
                f"{package} did not appear after install (type={preferred_type})."
            )
        log(f"Installed {package} via {preferred_type} path.")
    except Exception as e:
        log_warning(
            f"{preferred_type} install failed for {package}. "
            f"Falling back to source compilation. ({e})"
        )
        try:
            if already_installed and system == "Windows":
                _try_force_unload_package(package)
            utils.install_packages(
                StrVector((package,)),
                repos=StrVector(repos),
                # don't set type, let R manage this.
                Ncpus=cores,
            )
            installed_version = _get_r_pkg_version(package)
            if installed_version is None:
                raise RuntimeError(f"{package} did not appear after source install.")
            log(f"brmspy: Installed {package} from source.")
        except Exception as e2:
            log_error(f"Failed to install {package}.")
            raise e2

def _install_rpackage_deps(package: str):
    try:
        ro.r(f"""
            pkgs <- unique(unlist(
                tools::package_dependencies(
                    c("{package}"),
                    recursive = TRUE,
                    which = c("Depends", "Imports", "LinkingTo"),
                    db = available.packages()
                )
            ))

            to_install <- setdiff(pkgs, rownames(installed.packages()))
            if (length(to_install)) {{
                install.packages(to_install)
            }}
        """)
    except Exception as e:
        log_warning(str(e))
        return

def _build_cmstanr(tried_install_rtools: bool = False):
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
        log("Checking Windows toolchain (Rtools/cmdstanr)...")
        try:
            ro.r("cmdstanr::check_cmdstan_toolchain(fix = TRUE)")
        except Exception as e:
            log_error(f"Toolchain check failed: {e}")
            if tried_install_rtools:
                raise Exception(
                    "brmspy: Rtools auto-install failed. "
                    "Please install the matching Rtools version manually: "
                    "https://cran.r-project.org/bin/windows/Rtools/"
                )
            else:
                raise Exception(
                    "brmspy: Rtools missing and install_rtools = False. "
                    "Please set install_rtools = True or install the matching Rtools version manually: "
                    "https://cran.r-project.org/bin/windows/Rtools/"
                )

    ro.r(f"cmdstanr::install_cmdstan(cores = {cores}, overwrite = FALSE)")

def install_prebuilt(runtime_version="0.1.0", url: Optional[str] = None, bundle: Optional[str] = None, install_rtools: bool = False):
    """
    Install prebuilt brmspy runtime bundle for fast setup.
    
    Downloads and activates a precompiled runtime containing:
    - R packages (brms, cmdstanr, dependencies)
    - Compiled CmdStan binary
    - Complete environment ready for immediate use
    
    This reduces setup time from ~30 minutes to ~1 minute by avoiding
    compilation. Available for specific platform/R version combinations.

    This function reconfigures the running embedded R session to use an isolated 
    library environment from downloaded binaries. It does not mix or break the 
    default library tree already installed in the system.
    
    Parameters
    ----------
    runtime_version : str, default="0.1.0"
        Runtime schema version (not pip version)
    url : str, optional
        Custom URL for runtime bundle. If None, uses GitHub releases
    bundle : str, optional
        Local path to runtime bundle (.tar.gz or directory)
    install_rtools: bool, default=False
        Installs RTools (windows only) if they cant be found. 
        WARNING: Modifies system path and runs the full rtools installer.
    
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
    with LogTime("install_prebuilt"):
        _init()

        _forward_github_token_to_r()

        from brmspy.binaries import env

        if install_rtools:
            _install_rtools_for_current_r()

        if not env.can_use_prebuilt():
            raise RuntimeError(
                "Prebuilt binaries are not available for your system. "
                "Please install brms manually or in install_brms set use_prebuilt_binaries=False."
            )

        fingerprint = env.system_fingerprint()
        if url is None and bundle is None:
            url = f"https://github.com/kaitumisuuringute-keskus/brmspy/releases/download/runtime/brmspy-runtime-{runtime_version}-{fingerprint}.tar.gz"

        try:
            result = install_and_activate_runtime(
                url=url,
                bundle=bundle,
                runtime_version=runtime_version,
                activate=True,
                require_attestation=True
            )
            
            _get_brms()
            return result
        except Exception as e:
            log_warning(f"{e}")
            return False
    

def install_brms(
    brms_version: str = "latest",
    repo: Optional[str] = None,
    install_cmdstanr: bool = True,
    install_rstan: bool = False,
    cmdstanr_version: str = "latest",
    rstan_version: str = "latest",
    use_prebuilt_binaries: bool = False,
    install_rtools: bool = False
):
    """
    Install brms R package, optionally cmdstanr and CmdStan compiler, or rstan.
    
    Parameters
    ----------
    brms_version : str, default="latest"
        brms version: "latest", "2.23.0", or ">= 2.20.0"
    repo : str | None, default=None
        Extra CRAN repository URL
    install_cmdstanr : bool, default=True
        Whether to install cmdstanr and build CmdStan compiler
    install_rstan : bool, default=False
        Whether to install rstan (alternative to cmdstanr)
    cmdstanr_version : str, default="latest"
        cmdstanr version: "latest", "0.8.1", or ">= 0.8.0"
    rstan_version : str, default="latest"
        rstan version: "latest", "2.32.6", or ">= 2.32.0"
    use_prebuilt_binaries: bool, default=False
        Uses fully prebuilt binaries for cmdstanr and brms and their dependencies. 
        Ignores system R libraries and uses the latest brms and cmdstanr available 
        for your system. Requires R>=4 and might not be compatible with some older
        systems or missing toolchains. Can reduce setup time by 50x.
    install_rtools: bool, default=False
        Installs RTools (windows only) if they cant be found. 
        WARNING: Modifies system path and runs the full rtools installer. 
        Use with caution!
    
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
    with LogTime("install_brms"):
        _init()

        if use_prebuilt_binaries:
            if install_prebuilt(install_rtools=install_rtools):
                log("Setup complete! You're ready to use brmspy.")
                return

        if install_rtools:
            _install_rtools_for_current_r()
        
        _forward_github_token_to_r()

        log("Installing brms...")
        _install_rpackage("brms", version=brms_version, repos_extra=[repo])
        _install_rpackage_deps("brms")

        if install_cmdstanr:
            if platform.system() == "Windows":
                if _get_r_version() >= Version("4.5.0"):
                    log("R>=4.5 and OS is windows. Limiting cmdstanr version to >= 0.9")
                    if cmdstanr_version == "latest" or cmdstanr_version == "any" or not cmdstanr_version:
                        # cmdstanr <0.9 does not recognise rtools 45.
                        cmdstanr_version = ">= 0.9.0"

            log("Installing cmdstanr...")
            _install_rpackage("cmdstanr", version=cmdstanr_version, repos_extra=[
                "https://mc-stan.org/r-packages/",
                'https://stan-dev.r-universe.dev',
                repo
            ])
            _install_rpackage_deps("cmdstanr")
            log("Building cmdstanr...")
            _build_cmstanr(tried_install_rtools=install_rtools)

        if install_rstan:
            log("Installing rstan...")
            _install_rpackage("rstan", version=rstan_version, repos_extra=[repo])
            _install_rpackage_deps("rstan")

        _invalidate_singletons()
        # Import to mitigate lazy imports
        _get_brms()

        log("Setup complete! You're ready to use brmspy.")

        greet()



def get_brms_version() -> Optional[Version]:
    """
    Get installed brms R package version.
    
    Returns
    -------
    str
        Version object or None
    
    Raises
    ------
    ImportError
        If brms is not installed
    
    Examples
    --------

    ```python
    from brmspy import brms
    version = brms.get_brms_version()
    print(f"brms version: {version}")
    ```
    """
    return _get_r_pkg_version("brms")

