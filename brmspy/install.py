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
from brmspy.helpers.rtools import _install_rtools_for_current_r
from brmspy.helpers.singleton import _get_brms, _invalidate_singletons
from brmspy.helpers.rtools import _get_r_version

def _try_force_unload_package(package: str) -> None:
    """
    Try to unload and remove an R package as aggressively as possible.

    - Logs each step (start / ok / error) from inside R.
    - Does NOT raise on R-side failures; only logs them.
    - May still fail on Windows if DLLs are locked or dependencies keep it loaded.
    """
    r_code = f"""
      pkg <- "{package}"

      log_step <- function(step, expr) {{
        msg_prefix <- paste0("[unload:", pkg, "][", step, "] ")
        cat(msg_prefix, "start\\n", sep = "")
        res <- try(eval(expr), silent = TRUE)

        if (inherits(res, "try-error")) {{
          cond <- attr(res, "condition")
          if (!is.null(cond)) {{
            cat(msg_prefix, "ERROR: ", conditionMessage(cond), "\\n", sep = "")
          }} else {{
            cat(msg_prefix, "ERROR: ", as.character(res), "\\n", sep = "")
          }}
          FALSE
        }} else {{
          cat(msg_prefix, "ok\\n", sep = "")
          TRUE
        }}
      }}

      .unload_pkg <- function(pkg) {{
        

        # 1) Detach from search path
        log_step("detach_search", quote({{
          search_name <- paste0("package:", pkg)
          if (search_name %in% search()) {{
            detach(search_name, unload = TRUE, character.only = TRUE)
          }}
        }}))

        # 2) Unload namespace
        log_step("unloadNamespace", quote({{
          if (pkg %in% loadedNamespaces()) {{
            unloadNamespace(pkg)
          }}
        }}))

        # 3) pkgload (devtools-style unload)
        log_step("pkgload::unload", quote({{
          if (requireNamespace("pkgload", quietly = TRUE)) {{
            pkgload::unload(pkg)
          }}
        }}))

        # 4) DLL unload if still registered
        log_step("library.dynam.unload", quote({{
          dlls <- getLoadedDLLs()
          if (pkg %in% rownames(dlls)) {{
            dll_info <- dlls[[pkg]]
            dll_name <- dll_info[["name"]]
            libpath  <- dirname(dll_info[["path"]])
            library.dynam.unload(chname = dll_name,
                                 package = pkg,
                                 libpath = libpath)
          }}
        }}))

        # 5) Remove package from library if still installed
        log_step("remove.packages", quote({{
          ip <- installed.packages()
          if (pkg %in% rownames(ip)) {{
            remove.packages(pkg)
          }}
        }}))
      }}

      .unload_pkg(pkg)
    """

    try:
        print(f"[brmspy] Attempting aggressive unload of R package '{package}'")
        ro.r(r_code)
        print(f"[brmspy] Aggressive unload completed for '{package}'")
    except Exception as e:
        # rpy2 / transport-level failure – log, but don't kill caller
        print(f"[brmspy] Aggressive unload of '{package}' raised a Python/rpy2 exception: \n{e}")
    

def _init():
    # Set the CRAN mirror globally for this session. 
    # This prevents 'build-manifest.R' or subsequent installs from prompting for a mirror.
    ro.r('options(repos = c(CRAN = "https://cloud.r-project.org"))')

def _forward_github_token_to_r() -> None:
    """Forward GITHUB_PAT / GITHUB_TOKEN from Python env to R's Sys.getenv."""
    try:
        kwargs = {}
        pat = os.environ.get("GITHUB_PAT")
        token = os.environ.get("GITHUB_TOKEN")

        if not pat and not token:
            return
        
        r_setenv = cast(Callable, ro.r["Sys.setenv"])

        if pat:
            kwargs["GITHUB_PAT"] = pat
        elif token:
            kwargs["GITHUB_TOKEN"] = token

        if kwargs:
            r_setenv(**kwargs)
    except Exception as e:
        print(f"{e}")
        return

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


def _get_r_pkg_installed(package: str) -> bool:
    """
    Return True if `pkg` is installed in any library in .libPaths(),
    without loading the package/namespace.
    """
    expr = f"""
      suppressWarnings(suppressMessages(
        "{package}" %in% rownames(installed.packages())
      ))
    """
    return bool(cast(List, ro.r(expr))[0])


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
        print(
            f"brmspy: Installing {package} "
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

        print(
            f"brmspy: Installed {package} via remotes::install_version "
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
        print(f"brmspy: {package} {installed_version} already installed.")
        return

    print(f"brmspy: Installing {package} on {system} (Repos: {len(repos)})...")

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
        print(f"brmspy: Installed {package} via {preferred_type} path.")
    except Exception as e:
        print(
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
            print(f"brmspy: Installed {package} from source.")
        except Exception as e2:
            print(f"Failed to install {package}.")
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
        print(str(e))
        return

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
    _init()

    _forward_github_token_to_r()

    from brmspy.binaries import env
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

    _init()

    if use_prebuilt_binaries:
        if install_prebuilt():
            print("\nSetup complete! You're ready to use brmspy.")
            return
    
    _forward_github_token_to_r()

    print("Installing brms...")
    _install_rpackage("brms", version=brms_version, repos_extra=[repo])
    _install_rpackage_deps("brms")

    if install_cmdstanr:
        if platform.system() == "Windows":
            if _get_r_version() >= Version("4.5.0"):
                print("R>=4.5 and OS is windows. Limiting cmdstanr version to >= 0.9")
                if cmdstanr_version == "latest" or cmdstanr_version == "any" or not cmdstanr_version:
                    # cmdstanr <0.9 does not recognise rtools 45.
                    cmdstanr_version = ">= 0.9.0"

        print("Installing cmdstanr...")
        _install_rpackage("cmdstanr", version=cmdstanr_version, repos_extra=[
            "https://mc-stan.org/r-packages/",
            'https://stan-dev.r-universe.dev',
            repo
        ])
        _install_rpackage_deps("cmdstanr")
        print("Building cmdstanr...")
        _build_cmstanr()

    if install_rstan:
        print("Installing rstan...")
        _install_rpackage("rstan", version=rstan_version, repos_extra=[repo])
        _install_rpackage_deps("rstan")

    _invalidate_singletons()
    # Import to mitigate lazy imports
    _get_brms()

    print("\nSetup complete! You're ready to use brmspy.")