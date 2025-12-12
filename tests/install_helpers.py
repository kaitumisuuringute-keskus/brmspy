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
        refresh=0,
    )

    # Check it worked - now returns arviz InferenceData by default
    import arviz as az

    assert isinstance(model.idata, az.InferenceData)

    # Check key parameters exist
    param_names = list(cast(Any, model.idata).posterior.data_vars)
    assert any("b_zAge" in p for p in param_names)
    assert any("b_zBase" in p for p in param_names)
