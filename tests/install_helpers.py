

from typing import Any, cast


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
    from brmspy.runtime._activation import MANAGED_PACKAGES, _unload_managed_packages, _remove_managed_packages

    try:
        from brmspy.runtime._activation import _unload_managed_packages
        _unload_managed_packages()
    except Exception:
        pass
    
    for package in MANAGED_PACKAGES:
        try:
            if rpackages.isinstalled(package):
                ro.r(f'remove.packages("{package}")')
        except:
            pass
    
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

