"""
Runtime activation/deactivation. Mutates R environment ONLY.
Does NOT touch config - that's the caller's responsibility.
"""

from pathlib import Path
from typing import cast
from brmspy.helpers.log import log_warning
from brmspy.runtime import _manifest, _r_env, _r_packages, _state, _platform


MANAGED_PACKAGES = ("brms", "cmdstanr", "rstan")


def activate(runtime_path: Path) -> None:
    """
    Activate runtime by mutating R environment.
    
    Steps:
    1. Parse and validate manifest
    2. Store original R environment (if not already stored)
    3. Unload managed packages if loaded
    4. Set .libPaths() to runtime's Rlib/
    5. Set cmdstan path to runtime's cmdstan/
    6. Verify packages are loadable
    7. Invalidate package singletons
    
    Does NOT save to config. Caller handles that.
    
    On failure, attempts to restore original environment.
    """
    # Validate
    manifest = _manifest.parse_manifest(runtime_path / "manifest.json")
    if manifest is None:
        raise RuntimeError(f"Invalid manifest in {runtime_path}")
    
    _manifest.validate_manifest(manifest, _platform.system_fingerprint())
    
    # Capture original env (unless already captured from previous activation)
    if not _state.has_stored_env():
        original = _state.capture_current_env()
        _state.store_env(original)
    
    # Attempt activation with rollback on failure
    try:
        _unload_managed_packages()
        
        rlib = runtime_path / "Rlib"
        cmdstan = runtime_path / "cmdstan"

        rlib_posix = rlib.as_posix()
        cmdstan_posix = cmdstan.as_posix()
        
        _r_env.set_lib_paths([str(rlib_posix)])
        _r_env.set_cmdstan_path(str(cmdstan_posix))
        
        # Verify loadable
        _verify_runtime_loadable()
        
        _state.invalidate_packages()
        
    except Exception as e:
        # Rollback
        _rollback_to_stored_env()
        raise RuntimeError(f"Activation failed: {e}") from e


def deactivate() -> None:
    """
    Deactivate runtime by restoring original R environment.
    
    Does NOT clear config. Caller handles that.
    
    Raises:
        RuntimeError: If no stored environment to restore.
    """
    stored = _state.get_stored_env()
    if stored is None:
        raise RuntimeError("No runtime is currently active (no stored environment)")
    
    _unload_managed_packages()
    _r_env.set_lib_paths(stored.lib_paths)
    try:
        _r_env.set_cmdstan_path(stored.cmdstan_path)
    except Exception as e:
        log_warning(f"Failed to set_cmdstan_path to stored default ({stored.cmdstan_path}). Skipping! {e}")
    _state.clear_stored_env()
    _state.invalidate_packages()


def _unload_managed_packages() -> None:
    """Unload brms, cmdstanr, rstan if loaded."""
    for pkg in MANAGED_PACKAGES:
        if _r_env.is_namespace_loaded(pkg) or _r_env.is_package_attached(pkg):
            try:
                _r_env.unload_package(pkg)
            except Exception as e:
                log_warning(f"{e}")

def _remove_managed_packages() -> None:
    """removes brms, cmdstanr, rstan if loaded."""
    for pkg in MANAGED_PACKAGES:
        if _r_packages.is_package_installed(pkg):
            try:
                _r_packages.remove_package(pkg)
            except Exception as e:
                log_warning(f"{e}")

def _verify_runtime_loadable() -> None:
    """Verify brms and cmdstanr can be loaded."""
    from rpy2.robjects.packages import importr
    try:
        importr("brms")
        importr("cmdstanr")
    except Exception as e:
        raise RuntimeError(f"Cannot load runtime packages: {e}")


def _rollback_to_stored_env() -> None:
    """Restore original env on activation failure."""
    stored = _state.get_stored_env()
    if stored:
        try:
            _r_env.set_lib_paths(stored.lib_paths)
            _r_env.set_cmdstan_path(stored.cmdstan_path)
        except Exception:
            pass  # Best effort