import rpy2.robjects.packages as rpackages

from brmspy.helpers.log import log

_brms = None
_cmdstanr = None
_rstan = None
_posterior = None
_base = None

def _get_brms():
    """
    Lazy-load and cache brms R package and dependencies.
    
    First-time import loads brms, cmdstanr (optional), rstan (optional),
    posterior, and base R packages. Subsequent calls return cached
    instances. This singleton pattern ensures packages are imported
    only once per Python session.
    
    Returns
    -------
    rpy2 R package module
        Cached brms R package instance
    
    Raises
    ------
    ImportError
        If brms or posterior R packages are not installed.
        Error message includes installation instructions.
    
    Notes
    -----
    **Import Order and Fallbacks:**
    
    1. **cmdstanr** (optional): Stan backend, preferred for brms
       - If import fails, sets `_cmdstanr = None`
       - Not required if using rstan backend
    
    2. **rstan** (optional): Alternative Stan backend
       - If import fails, sets `_rstan = None`
       - Not required if using cmdstanr backend
    
    3. **posterior** (required): MCMC diagnostics and manipulation
       - Required by brms for posterior processing
       - Import failure raises ImportError
    
    4. **brms** (required): Main Bayesian modeling package
       - Import failure raises ImportError with installation help
    
    5. **base** (required): Base R functions
       - Always available with R installation
    
    **Singleton Pattern:**
    
    Uses module-level global variables to cache imports:
    - `_brms`: brms package
    - `_cmdstanr`: cmdstanr package (or None)
    - `_rstan`: rstan package (or None)
    - `_posterior`: posterior package
    - `_base`: base R package
    
    **First Import Message:**
    
    Prints "brmspy: Importing R libraries..." on first call to indicate
    R package loading (can take several seconds).
    
    Examples
    --------

    ```python
    from brmspy.helpers.singleton import _get_brms
    
    # First call: imports and caches R packages
    brms = _get_brms()  # Prints: "brmspy: Importing R libraries..."
    
    # Subsequent calls: returns cached instance (instant)
    brms_again = _get_brms()  # No import, uses cache
    assert brms is brms_again  # Same object
    ```

    See Also
    --------
    _get_base : Get base R package
    _get_cmdstanr : Get cmdstanr package (if available)
    _get_rstan : Get rstan package (if available)
    _invalidate_singletons : Clear cached imports
    brmspy.install.install_brms : Install brms R package
    """
    global _brms, _cmdstanr, _base, _posterior, _rstan
    if _brms is None:
        log("Importing R libraries...")
        try:
            try:
                _cmdstanr = rpackages.importr("cmdstanr")
            except Exception as e:
                _cmdstanr = None
            try:
                _rstan = rpackages.importr("rstan")
            except Exception as e:
                _rstan = None
            _posterior = rpackages.importr("posterior")
            _brms = rpackages.importr("brms")
            _base = rpackages.importr("base")
            log("R libraries imported!")
        except Exception as e:
            raise ImportError(
                "brms R package not found. Install it using:\n\n"
                "  import brmspy\n"
                "  brmspy.install_brms()  # for latest version\n\n"
                "Or install a specific version:\n"
                "  brmspy.install_brms(version='2.23.0')\n\n"
                "Or install manually in R:\n"
                "  install.packages('brms')\n"
            ) from e
    return _brms

def _get_base():
    """
    Get cached base R package instance.
    
    Returns the base R package imported by `_get_brms()`. Must be called
    after `_get_brms()` has been invoked at least once.
    
    Returns
    -------
    rpy2 R package module or None
        Cached base R package, or None if `_get_brms()` not yet called
    
    Notes
    -----
    This is a lightweight accessor for the base R package cached during
    brms import. The base package provides core R functions like `c()`,
    `list()`, `data.frame()`, etc.
    
    Always returns None until `_get_brms()` is called, as base is imported
    as part of the brms initialization process.
    
    Examples
    --------

    ```python
    from brmspy.helpers.singleton import _get_brms, _get_base
    
    # Must call _get_brms() first
    brms = _get_brms()
    
    # Now base is available
    base = _get_base()
    if base:
        # Use base R functions
        r_vec = base.c(1, 2, 3)
    ```

    See Also
    --------
    _get_brms : Main import function that caches base
    """
    global _base
    return _base

def _get_rstan():
    """
    Get cached rstan R package instance if available.
    
    Returns the rstan package if it was successfully imported by
    `_get_brms()`. rstan is an optional Stan backend alternative
    to cmdstanr.
    
    Returns
    -------
    rpy2 R package module or None
        Cached rstan package if import succeeded, None otherwise
    
    Notes
    -----
    **rstan vs cmdstanr:**
    
    Both are Stan backends for brms:
    - **rstan**: Older backend, integrated with brms longer
    - **cmdstanr**: Newer backend, generally faster and more flexible
    
    brms can use either backend. If both are available, cmdstanr is
    typically preferred.
    
    **Why rstan might be None:**
    
    - rstan not installed in R
    - rstan import failed during `_get_brms()`
    - User chose cmdstanr-only setup
    
    Examples
    --------

    ```python
    from brmspy.helpers.singleton import _get_brms, _get_rstan
    
    # Initialize imports
    brms = _get_brms()
    
    # Check if rstan available
    rstan = _get_rstan()
    if rstan:
        print("rstan backend available")
        # Could use rstan-specific functions
    else:
        print("rstan not available, using cmdstanr")
    ```

    See Also
    --------
    _get_brms : Main import function
    _get_cmdstanr : Get cmdstanr backend (alternative)
    """
    global _rstan
    return _rstan

def _get_cmdstanr():
    """
    Get cached cmdstanr R package instance if available.
    
    Returns the cmdstanr package if it was successfully imported by
    `_get_brms()`. cmdstanr is the preferred Stan backend for brms.
    
    Returns
    -------
    rpy2 R package module or None
        Cached cmdstanr package if import succeeded, None otherwise
    
    Notes
    -----
    **cmdstanr Backend:**
    
    cmdstanr is the modern Stan interface for R:
    - Uses CmdStan (Stan's command-line interface)
    - Generally faster than rstan
    - More flexible sampling options
    - Better integration with Stan ecosystem
    
    **Why cmdstanr might be None:**
    
    - cmdstanr not installed in R
    - cmdstanr import failed during `_get_brms()`
    - CmdStan not installed (cmdstanr requires CmdStan)
    - User chose rstan-only setup
    
    **Installation:**
    
    cmdstanr installation requires two steps:
    1. Install cmdstanr R package
    2. Install CmdStan (via cmdstanr::install_cmdstan())
    
    Examples
    --------

    ```python
    from brmspy.helpers.singleton import _get_brms, _get_cmdstanr
    
    # Initialize imports
    brms = _get_brms()
    
    # Check if cmdstanr available
    cmdstanr = _get_cmdstanr()
    if cmdstanr:
        print("cmdstanr backend available")
        # Could check CmdStan version
        version = cmdstanr.cmdstan_version()
    else:
        print("cmdstanr not available, using rstan")
    ```

    See Also
    --------
    _get_brms : Main import function
    _get_rstan : Get rstan backend (alternative)
    brmspy.install.install_cmdstan : Install CmdStan
    """
    global _cmdstanr
    return _cmdstanr

def _invalidate_singletons():
    """
    Clear all cached R package imports.
    
    Resets all singleton package caches to None, forcing fresh imports
    on next `_get_brms()` call. Useful for testing or after R environment
    changes (e.g., installing new packages, switching R versions).
    
    Notes
    -----
    **When to Use:**
    
    - Testing: Reset state between test runs
    - Package updates: Force reimport after R package installation
    - Runtime changes: After activating prebuilt runtime
    - Debugging: Clear potentially corrupted cached imports
    
    **What Gets Cleared:**
    
    - `_brms`: brms R package
    - `_cmdstanr`: cmdstanr R package
    - `_rstan`: rstan R package
    - `_base`: base R package
    - `_posterior`: posterior R package
    
    **Warning:**
    
    After calling this, all subsequent brms operations will trigger
    a fresh R package import, which takes several seconds.
    
    Examples
    --------

    ```python
    from brmspy.helpers.singleton import (
        _get_brms, _invalidate_singletons
    )
    
    # Use brms
    brms = _get_brms()
    
    # Install new R package version
    # (hypothetically via some other mechanism)
    
    # Force reimport to use new version
    _invalidate_singletons()
    brms = _get_brms()  # Fresh import with new version
    ```

    See Also
    --------
    _get_brms : Main import function that gets reset
    """
    global _brms, _cmdstanr, _base, _posterior, _rstan
    _brms = None
    _cmdstanr = None
    _base = None
    _posterior = None
    _rstan = None