"""
Runtime activation/deactivation. Mutates R environment ONLY.
Does NOT touch config - that's the caller's responsibility.
"""

from pathlib import Path
from typing import cast
from brmspy.helpers.log import log_warning
from brmspy.runtime import _manifest, _r_env, _state, _platform


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
        unload_all_non_base_packages()
        
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
    
    unload_all_non_base_packages()
    _r_env.set_lib_paths(stored.lib_paths)
    try:
        _r_env.set_cmdstan_path(stored.cmdstan_path)
    except Exception as e:
        log_warning(f"Failed to set_cmdstan_path to stored default ({stored.cmdstan_path}). Skipping! {e}")
    _state.clear_stored_env()
    _state.invalidate_packages()

def unload_managed_packages() -> None:
    """
    Unload brms, cmdstanr, rstan if loaded.
    
    Simple version that only targets managed packages.
    Use unload_all_non_base_packages() for comprehensive cleanup.
    """
    for pkg in MANAGED_PACKAGES:
        if _r_env.is_namespace_loaded(pkg) or _r_env.is_package_attached(pkg):
            try:
                _r_env.unload_package(pkg)
            except Exception as e:
                log_warning(f"Failed to unload {pkg}: {e}")


def unload_all_non_base_packages(max_passes: int = 3) -> None:
    """
    Unload all non-base R packages currently loaded in the environment.
    
    This keeps base and recommended packages (e.g., "stats", "utils", "graphics")
    but unloads everything else to ensure a clean R session.
    
    Uses multiple passes to handle dependency ordering - packages that depend
    on others must be unloaded first, and multiple passes naturally handles this
    without explicit dependency resolution.
    
    Parameters
    ----------
    max_passes : int, default=3
        Maximum number of unload passes. Each pass attempts to unload all
        non-base packages. Multiple passes handle dependency chains.
    
    Notes
    -----
    - Critical for Windows where loaded DLLs lock files
    - Runs GC between passes to release R object references
    - Failures are logged but don't raise exceptions
    """
    base_pkgs = _r_env.get_base_packages()
    
    for pass_num in range(max_passes):
        loaded_pkgs = set(_r_env.get_loaded_namespaces())
        to_unload = loaded_pkgs - base_pkgs
        
        if not to_unload:
            break
            
        for pkg in sorted(to_unload):
            try:
                _r_env.unload_package(pkg)
            except Exception as e:
                # Only log on final pass to reduce noise
                if pass_num == max_passes - 1:
                    log_warning(f"Failed to unload {pkg}: {e}")
        
        # GC between passes to release references
        _r_env.run_gc()

def remove_managed_packages() -> None:
    """
    Unload AND remove (uninstall) managed R packages.
    
    This is a destructive operation - use only in tests or
    when completely resetting the R environment.
    
    Order: unload all packages first (for clean DLL release),
    then remove managed packages.
    """
    # Full unload first - critical for Windows DLL release
    unload_all_non_base_packages()
    _r_env.run_gc()
    
    # Remove in dependency order (dependents first)
    packages_to_remove = ("brms", "cmdstanr", "rstan", "StanHeaders")
    
    for pkg in packages_to_remove:
        try:
            _r_env.remove_package(pkg)
        except Exception as e:
            log_warning(f"Failed to remove {pkg}: {e}")

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