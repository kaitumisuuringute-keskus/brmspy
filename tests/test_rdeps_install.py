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
    import rpy2.robjects as ro
    import rpy2.robjects.packages as rpackages
    import sys
    from brmspy.runtime._activation import MANAGED_PACKAGES, _unload_managed_packages

    try:
        from brmspy.runtime._activation import _unload_managed_packages
        _unload_managed_packages()
    except Exception:
        pass
    
    for package in MANAGED_PACKAGES:
        if rpackages.isinstalled(package):
            ro.r(f'remove.packages("{package}")')
    
    # Clear stored env to prevent state pollution from previous tests
    try:
        from brmspy.runtime._state import clear_stored_env
        clear_stored_env()
    except Exception:
        pass
    
    # since other tests might have imported brmspy already with global _brms singleton set,
    # we need to remove it from sys.modules first
    for name in list(sys.modules.keys()):
        if name.startswith("brmspy"):
            del sys.modules[name]



@pytest.mark.rdeps
class TestCrossplatformInstall:
    """Test brms installation and version checking on 3 major OS."""
    
    @pytest.mark.slow
    def test_brms_install(self):
        from brmspy import brms
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
        from brmspy import brms
        import rpy2.robjects.packages as rpackages
        _remove_deps()
        
        assert not rpackages.isinstalled("brms")
        assert not rpackages.isinstalled("cmdstanr")

        # Import after removal to ensure the library imports without brms installed
        from brmspy import brms
        from brmspy.runtime._state import get_brms

        brms.install_brms(use_prebuilt=True, install_rtools=False)

        assert rpackages.isinstalled("brms")
        assert rpackages.isinstalled("cmdstanr")

        _brms = get_brms()
        assert _brms is not None

        _fit_minimal_model(brms)

        # Use runtime API for deactivation
        from brmspy import runtime
        runtime.deactivate_brms()
