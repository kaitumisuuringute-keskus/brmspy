"""
Prediction helpers for brms models.

This module wraps brms prediction utilities and returns typed result objects that
contain both an ArviZ `InferenceData` view and the underlying R result.

Notes
-----
Executed inside the worker process that hosts the embedded R session.
"""

from collections.abc import Callable
import arviz as az
import typing
from typing import Any, Literal, cast, overload

import numpy as np
import pandas as pd

from ..helpers._rpy2._conversion import (
    _arviz_add_constant_data,
    _brmsfit_get_constant_data,
    _brmsfit_get_dims_and_coords,
    _brmsfit_get_observed_data,
    _brmsfit_get_posterior,
    _brmsfit_get_predict_generic,
    _brmsfit_get_response_names,
    _idata_add_resp_names_suffix,
    kwargs_r,
    py_to_r,
)
from ..types.brms_results import (
    FitResult,
    IDLogLikelihoodInsample,
    IDLogLikelihoodOutsample,
    IDObservedData,
    IDPosteriorPredictive,
    IDPredictions,
    IDResult,
    IDPosterior,
    ProxyListSexpVector,
)


def posterior(
    model: FitResult | ProxyListSexpVector, **kwargs
) -> IDResult[IDPosterior]:
    """
    Return posterior draws as idata.

    Wrapper around R ``posterior::as_draws_df()``.

    Parameters
    ----------
    model : FitResult
        Fitted model.
    **kwargs
        Forwarded to ``posterior::as_draws_df()``. e.g inc_warmup, regex, variable

    Returns
    -------
    PosteriorEpredResult
        Result containing `idata` (ArviZ `InferenceData`) and an underlying R handle.

    Examples
    --------
    ```python
    from brmspy import brms

    fit = brms.brm("y ~ x", data=df, chains=4)
    ep = brms.posterior(fit)

    ep.idata.posterior
    ```
    """
    model_r = py_to_r(model)
    kwargs = kwargs_r(kwargs)

    resp_names = _brmsfit_get_response_names(model_r)
    dims, coords = _brmsfit_get_dims_and_coords(model_r, resp_names=resp_names)

    result, r = _brmsfit_get_posterior(model_r, **kwargs)
    idata = az.from_dict(posterior=result, dims=dims, coords=coords)

    # Add constant data
    constant_data_dict = _brmsfit_get_constant_data(
        model_r, newdata=None, resp_names=resp_names
    )
    _arviz_add_constant_data(
        idata, constant_data_dict, "constant_data", obs_id=coords["obs_id"]
    )

    return IDResult(r=cast(ProxyListSexpVector, r), idata=cast(IDPosterior, idata))


def observed_data(model: FitResult | ProxyListSexpVector) -> IDResult[IDObservedData]:
    import rpy2.robjects as ro

    model_r = py_to_r(model)

    resp_names = _brmsfit_get_response_names(model_r)
    dims, coords = _brmsfit_get_dims_and_coords(model_r, resp_names=resp_names)

    result = _brmsfit_get_observed_data(model_r, resp_names=resp_names)
    r = cast(Any, ro.NULL)

    idata = az.from_dict(observed_data=result, coords=coords, dims=dims)
    idata = cast(IDObservedData, idata)

    # Add constant data
    constant_data_dict = _brmsfit_get_constant_data(
        model_r, newdata=None, resp_names=resp_names
    )
    _arviz_add_constant_data(idata, constant_data_dict, "constant_data")

    return IDResult(r=r, idata=idata)


@overload
def posterior_epred(
    model: FitResult | ProxyListSexpVector, newdata: Literal[None] = None
) -> IDResult[IDPosterior]: ...


@overload
def posterior_epred(
    model: FitResult | ProxyListSexpVector, newdata: pd.DataFrame
) -> IDResult[IDPredictions]: ...


def posterior_epred(
    model: FitResult | ProxyListSexpVector,
    newdata: pd.DataFrame | None = None,
    **kwargs,
) -> IDResult:
    """
    Compute expected posterior predictions (noise-free).

    Wrapper around R ``brms::posterior_epred()``. This returns draws of the
    expected value (typically on the response scale), without observation noise.

    Parameters
    ----------
    model : FitResult
        Fitted model.
    newdata : pandas.DataFrame or None, default=None
        New data for predictions. If ``None``, uses the training data.
    **kwargs
        Forwarded to ``brms::posterior_epred()``.

    Returns
    -------
    PosteriorEpredResult
        Result containing `idata` (ArviZ `InferenceData`) and an underlying R handle.

    See Also
    --------
    brms::posterior_epred : [R documentation](https://paulbuerkner.com/brms/reference/posterior_epred.brmsfit.html)

    Examples
    --------
    ```python
    from brmspy import brms

    fit = brms.brm("y ~ x", data=df, chains=4)
    ep = brms.posterior_epred(fit)

    ep.idata.posterior
    ```
    """
    model_r = py_to_r(model)
    data_r = py_to_r(newdata)
    kwargs = kwargs_r(kwargs)

    resp_names = _brmsfit_get_response_names(model_r)
    dims, coords = _brmsfit_get_dims_and_coords(
        model_r, resp_names=resp_names, newdata=newdata
    )

    result, r = _brmsfit_get_predict_generic(
        model_r,
        newdata=data_r,
        function="brms::posterior_epred",
        resp_names=resp_names,
        **kwargs,
    )

    if newdata is None:
        idata = az.from_dict(posterior=result, coords=coords, dims=dims)
        idata = cast(IDPosterior, idata)
    else:
        idata = az.from_dict(predictions=result, coords=coords, dims=dims)
        idata = cast(IDPredictions, idata)

    _idata_add_resp_names_suffix(idata, "_mean", resp_names)

    # Add constant data
    constant_data_dict = _brmsfit_get_constant_data(
        model_r, newdata=newdata, resp_names=resp_names
    )
    group_name = "constant_data" if newdata is None else "predictions_constant_data"
    _arviz_add_constant_data(idata, constant_data_dict, group_name)

    return IDResult(r=cast(ProxyListSexpVector, r), idata=idata)


@overload
def posterior_predict(
    model: FitResult | ProxyListSexpVector, newdata: Literal[None] = None, **kwargs
) -> IDResult[IDPosteriorPredictive]: ...


@overload
def posterior_predict(
    model: FitResult | ProxyListSexpVector, newdata: pd.DataFrame, **kwargs
) -> IDResult[IDPredictions]: ...


def posterior_predict(
    model: FitResult | ProxyListSexpVector,
    newdata: pd.DataFrame | None = None,
    **kwargs,
) -> IDResult:
    """
    Draw from the posterior predictive distribution (includes observation noise).

    Wrapper around R ``brms::posterior_predict()``.

    Parameters
    ----------
    model : FitResult
        Fitted model.
    newdata : pandas.DataFrame or None, default=None
        New data for predictions. If ``None``, uses the training data.
    **kwargs
        Forwarded to ``brms::posterior_predict()``.

    Returns
    -------
    PosteriorPredictResult
        Result containing `idata` (ArviZ `InferenceData`) and an underlying R handle.

    See Also
    --------
    brms::posterior_predict : [R documentation](https://paulbuerkner.com/brms/reference/posterior_predict.brmsfit.html)

    Examples
    --------
    ```python
    from brmspy import brms

    fit = brms.brm("y ~ x", data=df, chains=4)
    pp = brms.posterior_predict(fit)

    pp.idata.posterior_predictive
    ```
    """
    model_r = py_to_r(model)
    data_r = py_to_r(newdata)
    kwargs = kwargs_r(kwargs)

    resp_names = _brmsfit_get_response_names(model_r)
    dims, coords = _brmsfit_get_dims_and_coords(
        model_r, resp_names=resp_names, newdata=newdata
    )

    result, r = _brmsfit_get_predict_generic(
        model_r,
        newdata=data_r,
        function="brms::posterior_predict",
        resp_names=resp_names,
        **kwargs,
    )

    if newdata is None:
        idata = az.from_dict(
            posterior_predictive=result,
            dims=dims,
            coords=coords,
        )
        idata = cast(IDPosteriorPredictive, idata)
    else:
        idata = az.from_dict(
            predictions=result,
            dims=dims,
            coords=coords,
        )
        idata = cast(IDPredictions, idata)

    # Add constant data
    constant_data_dict = _brmsfit_get_constant_data(
        model_r, newdata=newdata, resp_names=resp_names
    )
    group_name = "constant_data" if newdata is None else "predictions_constant_data"
    _arviz_add_constant_data(idata, constant_data_dict, group_name)

    return IDResult(r=cast(ProxyListSexpVector, r), idata=idata)


@overload
def posterior_linpred(
    model: FitResult | ProxyListSexpVector, newdata: Literal[None] = None, **kwargs
) -> IDResult[IDPosterior]: ...


@overload
def posterior_linpred(
    model: FitResult | ProxyListSexpVector, newdata: pd.DataFrame, **kwargs
) -> IDResult[IDPredictions]: ...


def posterior_linpred(
    model: FitResult | ProxyListSexpVector,
    newdata: pd.DataFrame | None = None,
    **kwargs,
) -> IDResult:
    """
    Draw from the linear predictor.

    Wrapper around R ``brms::posterior_linpred()``. This typically returns draws
    on the link scale (before applying the inverse link), unless you pass
    ``transform=True``.

    Parameters
    ----------
    model : FitResult
        Fitted model.
    newdata : pandas.DataFrame or None, default=None
        New data for predictions. If ``None``, uses the training data.
    **kwargs
        Forwarded to ``brms::posterior_linpred()`` (commonly ``transform`` or ``ndraws``).

    Returns
    -------
    PosteriorLinpredResult
        Result containing `idata` (ArviZ `InferenceData`) and an underlying R handle.

    See Also
    --------
    brms::posterior_linpred : [R documentation](https://paulbuerkner.com/brms/reference/posterior_linpred.brmsfit.html)

    Examples
    --------
    ```python
    from brmspy import brms

    fit = brms.brm("y ~ x", data=df, chains=4)
    lp = brms.posterior_linpred(fit, transform=False)

    lp.idata.predictions
    ```
    """
    import rpy2.robjects as ro

    model_r = py_to_r(model)
    data_r = py_to_r(newdata)
    kwargs = kwargs_r(kwargs)

    resp_names = _brmsfit_get_response_names(model_r)
    dims, coords = _brmsfit_get_dims_and_coords(
        model_r, resp_names=resp_names, newdata=newdata
    )

    result, r = _brmsfit_get_predict_generic(
        model_r,
        newdata=data_r,
        function="brms::posterior_linpred",
        resp_names=resp_names,
        **kwargs,
    )

    if newdata is None:
        idata = az.from_dict(
            posterior=result,
            dims=dims,
            coords=coords,
        )
        idata = cast(IDPosterior, idata)
    else:
        idata = az.from_dict(
            predictions=result,
            dims=dims,
            coords=coords,
        )
        idata = cast(IDPredictions, idata)

    _idata_add_resp_names_suffix(idata, "_linpred", resp_names)

    # Add constant data
    constant_data_dict = _brmsfit_get_constant_data(
        model_r, newdata=newdata, resp_names=resp_names
    )
    group_name = "constant_data" if newdata is None else "predictions_constant_data"
    _arviz_add_constant_data(idata, constant_data_dict, group_name)

    return IDResult(r=cast(ProxyListSexpVector, r), idata=idata)


@overload
def log_lik(
    model: FitResult | ProxyListSexpVector, newdata: Literal[None] = None, **kwargs
) -> IDResult[IDLogLikelihoodInsample]: ...


@overload
def log_lik(
    model: FitResult | ProxyListSexpVector, newdata: pd.DataFrame, **kwargs
) -> IDResult[IDLogLikelihoodOutsample]: ...


def log_lik(
    model: FitResult | ProxyListSexpVector,
    newdata: pd.DataFrame | None = None,
    **kwargs,
) -> IDResult:
    """
    Compute pointwise log-likelihood draws.

    Wrapper around R ``brms::log_lik()``. The result is useful for LOO/WAIC via ArviZ.

    Parameters
    ----------
    model : FitResult
        Fitted model.
    newdata : pandas.DataFrame or None, default=None
        New data. If ``None``, uses the training data.
    **kwargs
        Forwarded to ``brms::log_lik()``.

    Returns
    -------
    LogLikResult
        Result containing `idata` (ArviZ `InferenceData`) and an underlying R handle.

    See Also
    --------
    brms::log_lik : [R documentation](https://paulbuerkner.com/brms/reference/log_lik.brmsfit.html)

    Examples
    --------
    ```python
    from brmspy import brms
    import arviz as az

    fit = brms.brm("y ~ x", data=df, chains=4)
    ll = brms.log_lik(fit)

    az.loo(ll.idata)
    ```
    """
    import rpy2.robjects as ro

    model_r = py_to_r(model)
    data_r = py_to_r(newdata)
    kwargs = kwargs_r(kwargs)

    resp_names = _brmsfit_get_response_names(model_r)
    dims, coords = _brmsfit_get_dims_and_coords(
        model_r, resp_names=resp_names, newdata=newdata
    )

    result, r = _brmsfit_get_predict_generic(
        model_r,
        newdata=data_r,
        function="brms::log_lik",
        resp_names=resp_names,
        **kwargs,
    )
    if newdata is None:
        idata = az.from_dict(log_likelihood=result, dims=dims, coords=coords)
        idata = cast(IDLogLikelihoodInsample, idata)
    else:
        idata = az.from_dict(log_likelihood=result, dims=dims, coords=coords)
        idata = cast(IDLogLikelihoodOutsample, idata)

    # Add constant data
    constant_data_dict = _brmsfit_get_constant_data(
        model_r, newdata=newdata, resp_names=resp_names
    )
    group_name = "constant_data" if newdata is None else "predictions_constant_data"
    _arviz_add_constant_data(idata, constant_data_dict, group_name)

    return IDResult(r=cast(ProxyListSexpVector, r), idata=idata)
