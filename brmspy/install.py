from typing import Optional, Union, cast, Tuple
from packaging.version import Version
import multiprocessing
import platform

import rpy2.robjects as ro
from rpy2.robjects.packages import importr
from rpy2.robjects.vectors import StrVector

from brmspy.helpers import _get_brms, _invalidate_singletons

def _parse_version_spec(spec: Optional[str]) -> Tuple[str, Optional[Version]]:
    """
    Parse a spec like '>=2.21.0', '==2.20.0', '2.21.0'.
    Returns (op, Version) where op in {'any','==','>=','<='}.
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
    if op == "any" or required is None:
        return True
    if op == "==":
        return installed == required
    if op == ">=":
        return installed >= required
    if op == "<=":
        return installed <= required
    # shouldn't get here
    return True

def _get_r_pkg_version(package: str) -> Optional[Version]:
    """
    Return installed R package version as packaging.Version, or None if not installed.
    """
    try:
        # utils::packageVersion("pkg") -> "x.y.z"
        v_str = cast(list, ro.r(f"as.character(utils::packageVersion('{package}'))"))[0]
        return Version(v_str)
    except Exception:
        return None
    
def _install_exact_version(package: str, version: Version, repos) -> None:
    """
    Install an exact package version via remotes::install_version.
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
    """Detects Linux distro and returns the Posit Binary Manager URL."""
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
            print(f"❌ Failed to install exact {package}=={required_version}: {e}")
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
        print(f"⚠️ {preferred_type} install failed. Falling back to source compilation. ({e})")
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
            print(f"❌ Failed to install {package}.")
            raise e2

def _build_cmstanr():
    cores = multiprocessing.cpu_count() - 1
    if cores > 4:
        cores = cores - 1

    # Load the library first
    ro.r("library(cmdstanr)")

    # Windows-specific fix: Ensure Rtools is found/configured
    if platform.system() == "Windows":
        print("brmspy: Checking Windows toolchain (Rtools)...")
        # 'fix=True' attempts to download/install or configure paths if missing
        try:
            ro.r("cmdstanr::check_cmdstan_toolchain(fix = TRUE)")
        except Exception as e:
            print(f"❌ Warning: Toolchain check failed: {e}")
            print("- You may need to manually install Rtools from: https://cran.r-project.org/bin/windows/Rtools/")

    ro.r(f'install_cmdstan(cores = {cores}, overwrite=FALSE)')



def install_brms(
    brms_version: str = "latest",
    repo: Optional[str] = None,
    install_cmdstanr: bool = True,
    install_rstan: bool = False,
    cmdstanr_version: str = "latest",
    rstan_version: str = "latest"
):
    """
    Install brms R package, optionally cmdstanr, and CmdStan compiler or rstan.
    
    Parameters
    ----------
    version : str, default="latest"
        brms version: "latest", "2.23.0", or ">=2.20.0"
    repo : str, default="https://cran.rstudio.com"
        CRAN repository URL
    install_cmdstan : bool, default=True
        Whether to install cmdstanr and CmdStan
    
    Examples
    --------
    >>> from brmspy import brms
    >>> brms.install_brms()
    >>> brms.install_brms(brms_version="2.23.0")
    >>> brms.install_brms(install_cmdstanr=False)
    """
    print("Installing brms...")
    _install_rpackage("brms", version=brms_version, repos_extra=[repo])

    if install_cmdstanr:
        print("Installin cmdstanr...")
        _install_rpackage("cmdstanr", version=cmdstanr_version, repos_extra=["https://mc-stan.org/r-packages/", repo])
        print("Building cmdstanr...")
        _build_cmstanr()

    if install_rstan:
        print("Installing rstan...")
        _install_rpackage("rstan", version=rstan_version, repos_extra=[repo])

    _invalidate_singletons()
    # Import to mitigate lazy imports
    _get_brms()

    print("\n✅ Setup complete! You're ready to use brmspy.")