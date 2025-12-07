
from pathlib import Path
import platform
from brmspy.helpers.log import log_warning
from brmspy.runtime import _r_packages, _storage
from brmspy.runtime._platform import system_fingerprint
from brmspy.runtime._types import RuntimeStatus, RuntimeManifest, SystemInfo
from packaging.version import Version

__all__ = [
    "install_brms", "install_runtime", "activate_runtime", "deactivate_runtime", "status", "get_brms_version",
    "RuntimeStatus", "RuntimeManifest", "SystemInfo",
]

# MUST be called as otherwise the environment gets stuck asking for one
_r_packages.set_cran_mirror()



def install_runtime(install_rtools: bool = False):
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
    install_rtools: bool, default=False
        Installs RTools (windows only) if they cant be found. 
        WARNING: Modifies system path and runs the full rtools installer.
    
    Returns
    -------
    bool
        Path if installation succeeded, None otherwise
    
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
    """
    return install_brms(
        use_prebuilt=True,
        install_rtools=install_rtools,
        activate=True,
    )


def get_brms_version() -> Version | None:
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
    version = status().brms_version
    if version is None:
        return None
    return Version(version)


def get_active_runtime() -> Path | None:
    """
    Get path to currently active prebuilt runtime.
    
    Returns
    -------
    Path or None
        Path to active runtime directory, or None if not configured
    
    Notes
    -----
    Returns None if:
    - No runtime configured in config file
    - Config file doesn't exist
    - Config file is corrupted
    
    Examples
    --------
    ```python
    from brmspy import get_active_runtime
    
    runtime_path = get_active_runtime()
    if runtime_path and runtime_path.exists():
        print(f"Active runtime: {runtime_path}")
    else:
        print("No active runtime configured")
    ```
    """
    return status().active_runtime


def install_brms(
    *,
    use_prebuilt: bool = False,
    install_rtools: bool = False,
    brms_version: str | None = None,
    cmdstanr_version: str | None = None,
    install_rstan: bool = True,
    install_cmdstanr: bool = True,
    rstan_version: str | None = None,
    activate: bool = True,
    **kwargs
) -> Path | None:
    """
    Install brms R package, optionally cmdstanr and CmdStan compiler, or rstan.

    WINDOWS WARNING: DO NOT run this if you have 
    
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
    use_prebuilt: bool, default=False
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
    brms.install_brms(use_prebuilt=True)
    """
    if "use_prebuilt_binaries" in kwargs:
        use_prebuilt = kwargs['use_prebuilt_binaries']
        log_warning("'use_prebuilt_binaries' is deprecated, please use 'use_prebuilt' instead")
    from brmspy.runtime import _install, _config, _activation, get_active_runtime, deactivate_runtime

    _r_packages.set_cran_mirror()
    
    if use_prebuilt:
        runtime_path = _install.install_runtime(install_rtools=install_rtools)
        
        if activate:
            _activation.activate(runtime_path)
            _config.set_active_runtime_path(runtime_path)
        
        return runtime_path
    else:
        if get_active_runtime():
            deactivate_runtime()

        _install.install_traditional(
            brms_version=brms_version,
            cmdstanr_version=cmdstanr_version,
            install_rstan=install_rstan,
            install_rtools=install_rtools,
            install_cmdstanr=install_cmdstanr,
            rstan_version=rstan_version
        )
        return None


def activate_runtime(runtime_path: Path | str | None = None) -> None:
    """
    Activate a runtime by mutating R environment.
    
    Parameters
    ----------
    runtime_path : Path or str or None, default=None
        Path to runtime directory. If None, uses last active runtime from config.
    
    Raises
    ------
    ValueError
        If runtime_path is None and no config exists.
    FileNotFoundError
        If runtime directory doesn't exist.
    RuntimeError
        If runtime structure is invalid or activation fails.
    
    Notes
    -----
    Side effects of activation:
    
    - Stores original R environment for later restoration
    - Unloads brms/cmdstanr/rstan if loaded
    - Sets R .libPaths() to runtime's Rlib/
    - Sets cmdstanr path to runtime's cmdstan/
    - Saves runtime_path to ~/.brmspy/config.json
    - Invalidates cached R package singletons
    """
    from brmspy.runtime import _config, _activation, _storage
    
    # Resolve path
    if runtime_path is None:
        runtime_path = _config.get_active_runtime_path()
        if runtime_path is None:
            runtime_path = find_local_runtime()

        if runtime_path is None:
            raise ValueError(
                "No runtime_path provided and no active runtime in config. "
                "Run install(use_prebuilt=True) first or provide a path."
            )
    else:
        runtime_path = Path(runtime_path)
    
    # Validate
    if not runtime_path.exists():
        raise FileNotFoundError(f"Runtime directory not found: {runtime_path}")
    if not _storage.is_runtime_dir(runtime_path):
        raise RuntimeError(f"Invalid runtime structure at: {runtime_path}")
    
    # Activate then persist
    _activation.activate(runtime_path)
    _config.set_active_runtime_path(runtime_path)


def deactivate_runtime() -> None:
    """
    Deactivate current runtime and restore original R environment.
    
    Raises
    ------
    RuntimeError
        If no runtime is currently active.
    
    Notes
    -----
    Side effects of deactivation:
    
    - Unloads brms/cmdstanr/rstan if loaded
    - Restores original .libPaths()
    - Restores original cmdstan path
    - Clears active_runtime from config
    - Invalidates cached R package singletons
    """
    from brmspy.runtime import _config, _activation, _state
    
    if not _state.has_stored_env():
        raise RuntimeError("No runtime is currently active")
    
    _activation.deactivate()
    _config.set_active_runtime_path(None)
    _state.invalidate_packages()


def status() -> RuntimeStatus:
    """
    Query current runtime status without side effects.
    
    Returns
    -------
    RuntimeStatus
        Dataclass with comprehensive state information including:
        
        - Active runtime path and activation state
        - System fingerprint and toolchain info
        - Prebuilt compatibility and availability
        - Installed brms/cmdstanr/rstan versions
    """
    from brmspy.runtime import _config, _platform, _state, _storage, _r_packages
    
    system = _platform.get_system_info()
    
    return RuntimeStatus(
        active_runtime=_config.get_active_runtime_path(),
        is_activated=_state.has_stored_env(),
        system=system,
        can_use_prebuilt=_platform.can_use_prebuilt(),
        prebuilt_available=_platform.is_prebuilt_available(system.fingerprint),
        compatibility_issues=tuple(_platform.get_compatibility_issues()),
        installed_runtimes=tuple(_storage.list_installed_runtimes()),
        brms_version=_r_packages.get_package_version("brms"),
        cmdstanr_version=_r_packages.get_package_version("cmdstanr"),
        rstan_version=_r_packages.get_package_version("rstan"),
    )

def find_local_runtime() -> Path | None:
    """
    Find an installed runtime matching the current system fingerprint.

    Uses ``system_fingerprint()`` to compute the current system identity and
    searches the local runtime store for a matching runtime directory.

    Returns
    -------
    Path or None
        Path to the matching runtime root directory if found,
        otherwise ``None``.

    Notes
    -----
    This function is a pure lookup:
    it does not install, activate, or modify any runtime state.
    """
    fingerprint = system_fingerprint()
    return _storage.find_runtime_by_fingerprint(fingerprint)


# === Internal: Auto-activation ===

def _autoload() -> None:
    """
    Restore last active runtime on module import.
    
    Notes
    -----
    This function is called automatically when the runtime module is imported.
    It attempts to restore the last active runtime from config. Failures are
    handled silently to avoid breaking imports.
    """
    from brmspy.runtime import _config, _activation, _storage
    
    path = _config.get_active_runtime_path()
    if path is None:
        return
    
    if not path.exists():
        log_warning(f"Failed to auto-activate saved runtime. Configured runtime no longer exists: {path}")
        _config.set_active_runtime_path(None)
        return
    
    if not _storage.is_runtime_dir(path):
        log_warning(f"Failed to auto-activate saved runtime. Configured runtime is invalid: {path}")
        _config.set_active_runtime_path(None)
        return

    if platform.system() == "Windows":
        log_warning("Autoloading previous prebuilt environment is disabled on windows. Please call activate()")

    try:
        _activation.activate(path)
    except Exception as e:
        log_warning(f"FFailed to auto-activate saved runtime {path}: {e}")
        _config.set_active_runtime_path(None)


    