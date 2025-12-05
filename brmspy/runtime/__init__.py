"""
brmspy runtime management.

Public API (4 functions only):
- install: Install brms and dependencies
- activate: Activate a runtime
- deactivate: Deactivate current runtime
- status: Query current runtime status
"""

from pathlib import Path
from brmspy.runtime._types import RuntimeStatus, RuntimeManifest, SystemInfo

__all__ = [
    "install", "activate", "deactivate", "status",
    "RuntimeStatus", "RuntimeManifest", "SystemInfo",
]


def install(
    *,
    use_prebuilt: bool = False,
    install_rtools: bool = False,
    brms_version: str | None = None,
    cmdstanr_version: str | None = None,
    install_rstan: bool = True,
    activate: bool = True,
) -> Path | None:
    """
    Install brms and dependencies.
    
    Args:
        use_prebuilt: If True, download prebuilt runtime bundle (~30 seconds).
                      If False, install R packages traditionally (~20-30 minutes).
        install_rtools: On Windows, install Rtools if not present.
        brms_version: Specific brms version (traditional install only).
        cmdstanr_version: Specific cmdstanr version (traditional install only).
        install_rstan: Install rstan package (traditional install only).
        activate: If True and use_prebuilt=True, activate runtime after install.
    
    Returns:
        Path to installed runtime if use_prebuilt=True, None otherwise.
    
    Raises:
        RuntimeError: If system requirements not met (with actionable message).
        ConnectionError: If download fails (prebuilt only).
    """
    from brmspy.runtime import _install, _config, _activation
    
    if use_prebuilt:
        runtime_path = _install.install_prebuilt(install_rtools=install_rtools)
        
        if activate:
            _activation.activate(runtime_path)
            _config.set_active_runtime_path(runtime_path)
        
        return runtime_path
    else:
        _install.install_traditional(
            brms_version=brms_version,
            cmdstanr_version=cmdstanr_version,
            install_rstan=install_rstan,
            install_rtools=install_rtools,
        )
        return None


def activate(runtime_path: Path | str | None = None) -> None:
    """
    Activate a runtime by mutating R environment.
    
    Args:
        runtime_path: Path to runtime directory. If None, uses last 
                     active runtime from config.
    
    Side effects:
        - Stores original R environment for later restoration
        - Unloads brms/cmdstanr/rstan if loaded
        - Sets R .libPaths() to runtime's Rlib/
        - Sets cmdstanr path to runtime's cmdstan/
        - Saves runtime_path to ~/.brmspy/config.json
        - Invalidates cached R package singletons
        
    Raises:
        ValueError: If runtime_path is None and no config exists.
        FileNotFoundError: If runtime directory doesn't exist.
        RuntimeError: If runtime structure is invalid or activation fails.
    """
    from brmspy.runtime import _config, _activation, _storage
    
    # Resolve path
    if runtime_path is None:
        runtime_path = _config.get_active_runtime_path()
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


def deactivate() -> None:
    """
    Deactivate current runtime, restore original R environment.
    
    Side effects:
        - Unloads brms/cmdstanr/rstan if loaded
        - Restores original .libPaths()
        - Restores original cmdstan path
        - Clears active_runtime from config
        - Invalidates cached R package singletons
        
    Raises:
        RuntimeError: If no runtime is currently active.
    """
    from brmspy.runtime import _config, _activation, _state
    
    if not _state.has_stored_env():
        raise RuntimeError("No runtime is currently active")
    
    _activation.deactivate()
    _config.set_active_runtime_path(None)


def status() -> RuntimeStatus:
    """
    Query current runtime status. No side effects.
    
    Returns:
        RuntimeStatus with comprehensive state information including:
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


# === Internal: Auto-activation ===

def _autoload() -> None:
    """
    Called on module import to restore last active runtime.
    Fails silently - must never break imports.
    """
    from brmspy.runtime import _config, _activation, _storage
    import logging
    
    path = _config.get_active_runtime_path()
    if path is None:
        return
    
    if not path.exists():
        logging.warning(f"Configured runtime no longer exists: {path}")
        _config.set_active_runtime_path(None)
        return
    
    if not _storage.is_runtime_dir(path):
        logging.warning(f"Configured runtime is invalid: {path}")
        _config.set_active_runtime_path(None)
        return
    
    try:
        _activation.activate(path)
    except Exception as e:
        logging.warning(f"Failed to auto-activate runtime: {e}")
        _config.set_active_runtime_path(None)