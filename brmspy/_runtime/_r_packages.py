"""
R package queries and installation. Stateless - no caching.
"""

import multiprocessing
import platform
from collections.abc import Callable
from typing import cast

from brmspy.helpers.log import log, log_error, log_warning

# === Queries ===


def get_package_version(name: str) -> str | None:
    """Get installed package version or None."""
    import rpy2.robjects as ro

    if not is_package_installed(name):
        return None
    try:
        expr = f"""
        v <- utils::packageDescription('{name}', fields = 'Version', lib.loc=.libPaths())
        if (is.na(v)) stop('Package not found')
        v
        """
        v_str = cast(list, ro.r(expr))[0]
        return str(v_str)
    except Exception:
        return None


def is_package_installed(name: str, lib_loc=None) -> bool:
    """Check if package is installed."""
    from rpy2.robjects.packages import isinstalled

    try:
        return isinstalled(name, lib_loc=lib_loc)
    except Exception:
        return False


# === Installation (traditional mode) ===


def set_cran_mirror(mirror: str | None = None) -> None:
    """
    Set CRAN mirror.
    Uses Posit Package Manager on Linux for binary packages.
    """
    import rpy2.robjects as ro

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
    repos_extra: str | list[str | None] | list[str] | None = None,
) -> None:
    import rpy2.robjects as ro
    from rpy2.robjects.packages import importr

    from brmspy._runtime._r_env import get_lib_paths, unload_package

    # Normalise special values that mean "latest / no constraint"
    if version is not None:
        v = version.strip()
        if v == "" or v.lower() in ("latest", "any"):
            version = None
        else:
            version = v

    set_cran_mirror()

    utils = importr("utils")
    system = platform.system()
    cores = multiprocessing.cpu_count()

    lib_path_py = [get_lib_paths()[0]]
    print("lib path is", lib_path_py)
    lib_path = ro.StrVector(lib_path_py)

    already_installed = is_package_installed(name, lib_loc=lib_path_py[0])

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
            f"Installing {name} "
            f"(version spec: {version!r}) via remotes::install_version()..."
        )

        # Ensure remotes is available
        ro.r(
            'if (!requireNamespace("remotes", quietly = TRUE)) '
            'install.packages("remotes", repos = "https://cloud.r-project.org")'
        )

        # Pass repo vector from Python into R
        ro.globalenv[".brmspy_repos"] = ro.StrVector(repos)

        # Escape double quotes in version spec just in case
        v_escaped = version.replace('"', '\\"')

        try:
            if already_installed and system == "Windows":
                unload_package(name)
            ro.r(
                f"remotes::install_version("
                f'package = "{name}", '
                f'version = "{v_escaped}", '
                f"repos = .brmspy_repos)"
            )
        finally:
            # Clean up
            del ro.globalenv[".brmspy_repos"]

        installed_version = get_package_version(name)
        if installed_version is None:
            raise RuntimeError(
                f"{name} did not appear after remotes::install_version('{version}')."
            )

        log(
            f"Installed {name} via remotes::install_version "
            f"(installed: {installed_version})."
        )
        return

    # ------------------------------------------------------------------
    # BRANCH 2: no version spec -> "latest" from repos via install.packages
    # ------------------------------------------------------------------
    installed_version = None
    try:
        if already_installed:
            installed_version = get_package_version(name)
    except Exception:
        installed_version = None

    if installed_version is not None:
        log(f"{name} {installed_version} already installed.")
        return

    log(f"Installing {name} on {system} (Repos: {len(repos)})...")

    try:
        # Primary Attempt (Fast Binary / P3M)
        if already_installed and system == "Windows":
            unload_package(name)
        utils.install_packages(
            ro.StrVector((name,)),
            repos=ro.StrVector(repos),
            lib=lib_path,
            type=preferred_type,
            Ncpus=cores,
        )
        installed_version = get_package_version(name)
        if installed_version is None:
            raise RuntimeError(
                f"{name} did not appear after install (type={preferred_type})."
            )
        log(f"Installed {name} via {preferred_type} path.")
    except Exception as e:
        log_warning(
            f"{preferred_type} install failed for {name}. "
            f"Falling back to source compilation. ({e})"
        )
        try:
            if already_installed and system == "Windows":
                unload_package(name)
            utils.install_packages(
                ro.StrVector((name,)),
                repos=ro.StrVector(repos),
                # don't set type, let R manage this.
                lib=lib_path,
                Ncpus=cores,
            )
            installed_version = get_package_version(name)
            if installed_version is None:
                raise RuntimeError(f"{name} did not appear after source install.")
            log(f"brmspy: Installed {name} from source.")
        except Exception as e2:
            log_error(f"Failed to install {name}: {e2}")
            raise e2


def install_package_deps(
    name: str,
    include_suggests: bool = False,
    repos_extra: str | list[str | None] | list[str] | None = None,
) -> None:
    """Install dependencies of an R package."""
    import rpy2.robjects as ro

    set_cran_mirror()

    which_deps = ro.StrVector(["Depends", "Imports", "LinkingTo"])
    if include_suggests:
        which_deps = ro.StrVector(["Depends", "Imports", "LinkingTo", "Suggests"])

    ncpus = multiprocessing.cpu_count() - 1
    ncpus = max(1, ncpus)

    repos: list[str] = ["https://cloud.r-project.org"]  # good default mirror

    if repos_extra:
        if isinstance(repos_extra, list):
            for _r in repos_extra:
                if isinstance(_r, str) and _r not in repos:
                    repos.append(_r)
        elif repos_extra not in repos:
            repos.append(repos_extra)

    try:
        cast(
            Callable,
            ro.r(
                """
        function (which_deps, name, ncpus, repos) {
            pkgs <- unique(unlist(
                tools::package_dependencies(
                    name,
                    recursive = TRUE,
                    which = which_deps,
                    db = available.packages()
                )
            ))
            
            to_install <- setdiff(pkgs, rownames(installed.packages(lib.loc = .libPaths(), noCache = TRUE)))
            if (length(to_install)) {
                install.packages(to_install, Ncpus = ncpus, repos = repos, lib = .libPaths()[1L])
            }
        }
        """
            ),
        )(which_deps, ro.StrVector([name]), ncpus, ro.StrVector(repos))
    except Exception as e:
        log_warning(str(e))
        return


def build_cmdstan(cores: int | None = None) -> None:
    """Build CmdStan via cmdstanr::install_cmdstan()."""
    import rpy2.robjects as ro

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


def remove_package(name: str) -> bool:
    import rpy2.robjects as ro

    r_code = f"""
    (function(pkg) {{
        removed <- FALSE
        libs <- .libPaths()
        
        for (lib in libs) {{
            pkg_path <- file.path(lib, pkg)
            if (dir.exists(pkg_path)) {{
                tryCatch({{
                    # Use normalized path for cross-platform safety
                    lib_norm <- normalizePath(lib, winslash = "/", mustWork = FALSE)
                    suppressWarnings(remove.packages(pkg, lib = lib_norm))
                    removed <- TRUE
                }}, error = function(e) {{
                    # Windows fallback: try direct deletion if DLLs are unloaded
                    if (.Platform$OS.type == "windows") {{
                        tryCatch({{
                            unlink(pkg_path, recursive = TRUE, force = TRUE)
                            removed <- TRUE
                        }}, error = function(e2) NULL)
                    }}
                }})
            }}
        }}
        
        # Check if actually removed
        return(!dir.exists(file.path(.libPaths()[1], pkg)))
    }})('{name}')
    """

    try:
        result = cast(list, ro.r(r_code))
        return str(result[0]).lower().strip() == "true"
    except Exception:
        return False
