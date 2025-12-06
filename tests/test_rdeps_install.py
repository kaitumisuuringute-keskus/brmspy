"""
Dep installation tests for Windows, Ubuntu, and macOS

These tests are DESTRUCTIVE for the R environment.
DO NOT run locally!

These tests ONLY run within github actions, as running 
all 3 major platform images from a single local machine 
is both legally and technically difficult.
"""

from typing import Any, cast
import pytest
import sys
import gc


def _fit_minimal_model(brms):
    """Run a very small model to verify the installation."""
    epilepsy = brms.get_brms_data("epilepsy")
    
    # Fit model (with reduced iterations for testing)
    model = brms.fit(
        formula="count ~ zAge + zBase * Trt + (1|patient)",
        data=epilepsy,
        family="poisson",
        iter=100,
        warmup=50,
        chains=2,
        silent=2,
        refresh=0
    )
    
    # Check it worked - now returns arviz InferenceData by default
    import arviz as az
    assert isinstance(model.idata, az.InferenceData)
    
    # Check key parameters exist
    param_names = list(cast(Any, model.idata).posterior.data_vars)
    assert any('b_zAge' in p for p in param_names)
    assert any('b_zBase' in p for p in param_names)



def _remove_deps():
    """
    Remove all managed R packages in an OS-safe and order-safe way.
    
    Handles:
    - Cross-platform path/DLL differences (especially Windows)
    - Package dependency ordering via multiple passes
    - Partial/failed states
    - Works regardless of what's currently loaded
    
    This is DESTRUCTIVE - only use in CI tests!
    """
    # 1. Clear Python modules FIRST to release any R object references
    #    Must happen before importing brmspy components
    from brmspy.runtime._state import invalidate_packages
    invalidate_packages()

    # 2. Now safe to import and use library functions
    from brmspy.runtime._activation import (
        _unload_managed_packages, _remove_managed_packages
        
    )
    from brmspy.runtime._r_env import run_gc
    
    # 3. Full unload and remove using library functions
    try:
        _unload_managed_packages()
    except Exception:
        pass
    
    run_gc()
    
    try:
        _remove_managed_packages()
    except Exception:
        pass
    
    run_gc()
    
    # 4. Final Python module cleanup (in case re-imported during above)
    from brmspy.runtime._state import invalidate_packages
    invalidate_packages()



@pytest.mark.rdeps
class TestCrossplatformInstall:
    """Test brms installation and version checking on 3 major OS."""
    
    @pytest.mark.slow
    def test_brms_install(self):
        import rpy2.robjects.packages as rpackages
        _remove_deps()
        
        assert not rpackages.isinstalled("brms")
        assert not rpackages.isinstalled("cmdstanr")

        # Import after removal to ensure the library imports without brms installed
        from brmspy import brms
        from brmspy.runtime._state import get_brms

        brms.install_brms(use_prebuilt=False)

        assert rpackages.isinstalled("brms")
        assert rpackages.isinstalled("cmdstanr")

        _brms = get_brms()
        assert _brms is not None

        _fit_minimal_model(brms)
    
    @pytest.mark.slow
    def test_brms_install_prebuilt(self):
        import rpy2.robjects.packages as rpackages
        _remove_deps()
        
        assert not rpackages.isinstalled("brms")
        assert not rpackages.isinstalled("cmdstanr")

        # Import after removal to ensure the library imports without brms installed
        from brmspy import brms
        from brmspy.runtime._state import get_brms

        brms.install_brms(use_prebuilt=True, install_rtools=True)

        assert rpackages.isinstalled("brms")
        assert rpackages.isinstalled("cmdstanr")

        _brms = get_brms()
        assert _brms is not None

        _fit_minimal_model(brms)

        # Use runtime API for deactivation
        from brmspy import runtime
        runtime.deactivate()
