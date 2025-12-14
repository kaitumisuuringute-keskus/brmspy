"""
Prediction helpers for brms models.

This module wraps brms prediction utilities and returns typed result objects that
contain both an ArviZ `InferenceData` view and the underlying R result.

Notes
-----
Executed inside the worker process that hosts the embedded R session.
"""

import typing

import pandas as pd

from ..helpers._rpy2._conversion import (
    brms_epred_to_idata,
    brms_linpred_to_idata,
    brms_log_lik_to_idata,
    brms_predict_to_idata,
    kwargs_r,
    py_to_r,
)
from ..types.brms_results import (
    FitResult,
    IDEpred,
    IDLinpred,
    IDLogLik,
    IDPredict,
    LogLikResult,
    PosteriorEpredResult,
    PosteriorLinpredResult,
    PosteriorPredictResult,
)


def posterior_epred(
    model: FitResult, newdata: pd.DataFrame | None = None, **kwargs
) -> PosteriorEpredResult:
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

    ep.idata.predictions
    ```
    """
    import rpy2.robjects as ro

    m = model.r
    data_r = py_to_r(newdata)
    kwargs = kwargs_r(kwargs)

    # Get R function explicitly
    r_posterior_epred = typing.cast(typing.Callable, ro.r("brms::posterior_epred"))

    # Call with proper argument names (object instead of model)
    r = r_posterior_epred(m, newdata=data_r, **kwargs)
    idata = brms_epred_to_idata(r, model.r, newdata=newdata)
    idata = typing.cast(IDEpred, idata)

    return PosteriorEpredResult(r=r, idata=idata)


def posterior_predict(
    model: FitResult, newdata: pd.DataFrame | None = None, **kwargs
) -> PosteriorPredictResult:
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
    import rpy2.robjects as ro

    m = model.r

    data_r = py_to_r(newdata)
    kwargs = kwargs_r(kwargs)

    # Get R function explicitly
    r_posterior_predict = typing.cast(typing.Callable, ro.r("brms::posterior_predict"))

    # Call with proper arguments
    if newdata is not None:
        r = r_posterior_predict(m, newdata=data_r, **kwargs)
    else:
        r = r_posterior_predict(m, **kwargs)

    idata = brms_predict_to_idata(r, model.r, newdata=newdata)
    idata = typing.cast(IDPredict, idata)

    return PosteriorPredictResult(r=r, idata=idata)


def posterior_linpred(
    model: FitResult, newdata: pd.DataFrame | None = None, **kwargs
) -> PosteriorLinpredResult:
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

    m = model.r

    data_r = py_to_r(newdata)
    kwargs = kwargs_r(kwargs)

    # Get R function explicitly
    r_posterior_linpred = typing.cast(typing.Callable, ro.r("brms::posterior_linpred"))

    # Call with proper arguments
    if newdata is not None:
        r = r_posterior_linpred(m, newdata=data_r, **kwargs)
    else:
        r = r_posterior_linpred(m, **kwargs)

    idata = brms_linpred_to_idata(r, model.r, newdata=newdata)
    idata = typing.cast(IDLinpred, idata)

    return PosteriorLinpredResult(r=r, idata=idata)


def log_lik(
    model: FitResult, newdata: pd.DataFrame | None = None, **kwargs
) -> LogLikResult:
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

    m = model.r

    data_r = py_to_r(newdata)
    kwargs = kwargs_r(kwargs)

    # Get R function explicitly
    r_log_lik = typing.cast(typing.Callable, ro.r("brms::log_lik"))

    # Call with proper arguments
    if newdata is not None:
        r = r_log_lik(m, newdata=data_r, **kwargs)
    else:
        r = r_log_lik(m, **kwargs)

    idata = brms_log_lik_to_idata(r, model.r, newdata=newdata)
    idata = typing.cast(IDLogLik, idata)

    return LogLikResult(r=r, idata=idata)
