

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
    from brmspy.runtime._activation import _unload_managed_packages, _remove_managed_packages
    from brmspy.runtime._state import invalidate_packages
    from brmspy.runtime import get_active_runtime, deactivate_runtime
    from brmspy.runtime._activation import _unload_managed_packages

    if get_active_runtime():
        deactivate_runtime()

    try:
        _unload_managed_packages()
    except Exception:
        pass
    
    try:
        _remove_managed_packages()
    except Exception:
        pass

    invalidate_packages()

