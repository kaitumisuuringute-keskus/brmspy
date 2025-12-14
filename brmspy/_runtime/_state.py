"""
All singletons in one place: R packages and stored environment.
This is the ONLY module with global mutable state.
"""

from typing import Any

from brmspy._runtime import _r_env
from brmspy.types.runtime import StoredEnv

# === Package singletons (lazy-imported) ===

_brms: Any = None
_cmdstanr: Any = None
_rstan: Any = None
_base: Any = None
_posterior: Any = None


def get_brms() -> Any:
    """Get brms R package, importing on first access."""
    global _brms
    if _brms is None:
        try:
            from rpy2.robjects.packages import importr
            import rpy2.robjects as ro

            _brms = importr("brms")
            ro.r("library(brms)")
        except Exception as e:
            raise ImportError(
                "brms R package not found. Install it using:\n\n"
                "  import brmspy\n"
                "  brmspy.install_brms(use_prebuilt=True)  # for prebuilt binaries\n\n"
                "Or install from source:\n"
                "  brmspy.install_brms()\n"
            ) from e
    return _brms


def get_cmdstanr() -> Any | None:
    """Get cmdstanr R package or None if not available."""
    global _cmdstanr
    if _cmdstanr is None:
        try:
            from rpy2.robjects.packages import importr

            _cmdstanr = importr("cmdstanr")
        except Exception:
            pass
    return _cmdstanr


def get_rstan() -> Any | None:
    """Get rstan R package or None if not available."""
    global _rstan
    if _rstan is None:
        try:
            from rpy2.robjects.packages import importr

            _rstan = importr("rstan")
        except Exception:
            pass
    return _rstan


def get_base() -> Any:
    """Get base R package."""
    global _base
    if _base is None:
        from rpy2.robjects.packages import importr

        _base = importr("base")
    return _base


def get_posterior() -> Any:
    """Get posterior R package."""
    global _posterior
    if _posterior is None:
        from rpy2.robjects.packages import importr

        _posterior = importr("posterior")
    return _posterior


def invalidate_packages() -> None:
    """Clear all cached package singletons."""
    global _brms, _cmdstanr, _rstan, _base, _posterior
    _brms = None
    _cmdstanr = None
    _rstan = None
    _base = None
    _posterior = None


# === Stored environment (for deactivation) ===

_stored_env: StoredEnv | None = None


def capture_current_env() -> StoredEnv:
    """Capture current R environment (lib_paths, cmdstan_path)."""
    return StoredEnv(
        lib_paths=_r_env.get_lib_paths(),
        cmdstan_path=_r_env.get_cmdstan_path(),
    )


def store_env(env: StoredEnv) -> None:
    """Store environment for later restoration."""
    if ".brmspy/runtime/" in env.lib_paths or ".brmspy\\runtime\\" in env.lib_paths:
        return
    global _stored_env
    _stored_env = env


def get_stored_env() -> StoredEnv | None:
    """Get stored environment or None."""
    return _stored_env


def clear_stored_env() -> None:
    """Clear stored environment."""
    global _stored_env
    _stored_env = None


def has_stored_env() -> bool:
    """Check if environment is stored (i.e., runtime is active)."""
    return _stored_env is not None
