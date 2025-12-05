"""
R package queries and installation. Stateless - no caching.
"""

import platform
import multiprocessing
from typing import List, Optional, Union, cast

import rpy2.robjects as ro
from rpy2.robjects.packages import importr
from rpy2.robjects.vectors import StrVector


# === Queries ===

def get_package_version(name: str) -> str | None:
    """Get installed package version or None."""
    try:
        expr = f"""
        v <- utils::packageDescription('{name}', fields = 'Version')
        if (is.na(v)) stop('Package not found')
        v
        """
        v_str = cast(List, ro.r(expr))[0]
        return str(v_str)
    except Exception:
        return None


def is_package_installed(name: str) -> bool:
    """Check if package is installed."""
    from rpy2.robjects.packages import isinstalled
    try:
        return isinstalled(name)
    except Exception:
        return False


# === Installation (traditional mode) ===

def set_cran_mirror(mirror: str | None = None) -> None:
    """
    Set CRAN mirror. 
    Uses Posit Package Manager on Linux for binary packages.
    """
    if mirror is None:
        mirror = "https://cloud.r-project.org"
    ro.r(f'options(repos = c(CRAN = "{mirror}"))')


def _get_linux_repo() -> str:
    """Get Posit Package Manager URL for Linux binaries."""
    try:
        with open("/etc/os-release") as f:
            lines = f.readlines()
        
        codename = "jammy"  # Default fallback (Ubuntu 22.04)
        for line in lines:
            if line.startswith("VERSION_CODENAME="):
                codename = line.strip().split("=")[1].strip('"')
                break
        
        return f"https://packagemanager.posit.co/cran/__linux__/{codename}/latest"
    except FileNotFoundError:
        return "https://packagemanager.posit.co/cran/__linux__/jammy/latest"


def install_package(
    name: str,
    version: str | None = None,
    repos_extra: Optional[Union[str, List[Optional[str]], List[str]]] = None
) -> None:
    """
    Install single R package.
    Uses remotes::install_version if version specified,
    otherwise utils::install.packages.
    """
    # Normalize special values
    if version is not None:
        v = version.strip()
        if v == "" or v.lower() in ("latest", "any"):
            version = None
        else:
            version = v
    
    utils = importr("utils")
    system = platform.system()
    cores = multiprocessing.cpu_count()
    
    already_installed = is_package_installed(name)
    
    repos: list[str] = ["https://cloud.r-project.org"]

    if repos_extra:
        if isinstance(repos_extra, list):
            for _r in repos_extra:
                if isinstance(_r, str) and _r not in repos:
                    repos.append(_r)
        elif repos_extra and repos_extra not in repos:
            repos.append(repos_extra)
    
    if system == "Linux":
        binary_repo = _get_linux_repo()
        repos.insert(0, binary_repo)
        preferred_type = "source"
    else:
        preferred_type = "binary"
    
    # Version specified -> use remotes
    if version is not None:
        # Ensure remotes is available
        ro.r(
            'if (!requireNamespace("remotes", quietly = TRUE)) '
            'install.packages("remotes", repos = "https://cloud.r-project.org")'
        )
        
        ro.globalenv[".brmspy_repos"] = StrVector(repos)
        v_escaped = version.replace('"', '\\"')
        
        try:
            if already_installed and system == "Windows":
                from brmspy.runtime._r_env import unload_package
                unload_package(name)
            
            ro.r(
                f'remotes::install_version('
                f'package = "{name}", '
                f'version = "{v_escaped}", '
                f'repos = .brmspy_repos)'
            )
        finally:
            del ro.globalenv[".brmspy_repos"]
        
        return
    
    # No version -> install latest
    if is_package_installed(name):
        return
    
    try:
        if already_installed and system == "Windows":
            from brmspy.runtime._r_env import unload_package
            unload_package(name)
        
        utils.install_packages(
            StrVector((name,)),
            repos=StrVector(repos),
            type=preferred_type,
            Ncpus=cores,
        )
    except Exception:
        # Fallback to source
        utils.install_packages(
            StrVector((name,)),
            repos=StrVector(repos),
            Ncpus=cores,
        )


def install_package_deps(name: str, include_suggests: bool = False) -> None:
    """Install dependencies of an R package."""
    which_deps = 'c("Depends", "Imports", "LinkingTo")'
    if include_suggests:
        which_deps = 'c("Depends", "Imports", "LinkingTo", "Suggests")'
    
    ro.r(f"""
        pkgs <- unique(unlist(
            tools::package_dependencies(
                c("{name}"),
                recursive = TRUE,
                which = {which_deps},
                db = available.packages()
            )
        ))
        
        to_install <- setdiff(pkgs, rownames(installed.packages()))
        if (length(to_install)) {{
            install.packages(to_install)
        }}
    """)


def build_cmdstan(cores: int | None = None) -> None:
    """Build CmdStan via cmdstanr::install_cmdstan()."""
    if cores is None:
        cores = multiprocessing.cpu_count()
        if cores > 4:
            cores -= 1
    
    ro.r("library(cmdstanr)")
    
    if platform.system() == "Windows":
        try:
            ro.r("cmdstanr::check_cmdstan_toolchain(fix = TRUE)")
        except Exception as e:
            raise RuntimeError(
                "Windows toolchain check failed. "
                "Please install Rtools from https://cran.r-project.org/bin/windows/Rtools/ "
                "or run install(install_rtools=True)"
            ) from e
    
    ro.r(f"cmdstanr::install_cmdstan(cores = {cores}, overwrite = FALSE)")