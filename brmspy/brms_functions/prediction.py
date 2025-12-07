import typing
import pandas as pd

from ..runtime._state import get_brms
from ..helpers.conversion import (
    brms_linpred_to_idata, brms_log_lik_to_idata, brms_epred_to_idata, brms_predict_to_idata,
    kwargs_r, py_to_r
)
from ..types import (
    FitResult, IDEpred, IDLinpred, IDLogLik,
    IDPredict, LogLikResult, PosteriorEpredResult, PosteriorLinpredResult, PosteriorPredictResult
)


def posterior_epred(model: FitResult, newdata: pd.DataFrame, **kwargs) -> PosteriorEpredResult:
    """
    Compute expected value of posterior predictive distribution.
    
    Calls brms::posterior_epred() to get E[Y|data] without observation noise.

    [BRMS documentation and parameters](https://paulbuerkner.com/brms/reference/posterior_epred.brmsfit.html)
    
    Parameters
    ----------
    model : FitResult
        Fitted model from fit()
    newdata : pd.DataFrame
        Data for predictions
    **kwargs
        Additional arguments to brms::posterior_epred()
    
    Returns
    -------
    PosteriorEpredResult
        Object with .idata and .r attributes
    """
    import rpy2.robjects as ro
    get_brms()  # Ensure brms is loaded
    m = model.r
    data_r = py_to_r(newdata)
    kwargs = kwargs_r(kwargs)

    # Get R function explicitly
    r_posterior_epred = typing.cast(typing.Callable, ro.r('brms::posterior_epred'))
    
    # Call with proper argument names (object instead of model)
    r = r_posterior_epred(m, newdata=data_r, **kwargs)
    idata = brms_epred_to_idata(r, model.r, newdata=newdata)
    idata = typing.cast(IDEpred, idata)

    return PosteriorEpredResult(
        r=r, idata=idata
    )

def posterior_predict(model: FitResult, newdata: typing.Optional[pd.DataFrame] = None, **kwargs) -> PosteriorPredictResult:
    """
    Generate posterior predictive samples with observation noise.
    
    Calls brms::posterior_predict() to get samples of Y_new|data.

    [BRMS documentation and parameters](https://paulbuerkner.com/brms/reference/posterior_predict.brmsfit.html)
    
    Parameters
    ----------
    model : FitResult
        Fitted model from fit()
    newdata : pd.DataFrame, optional
        Data for predictions. If None, uses original data
    **kwargs
        Additional arguments to brms::posterior_predict()
    
    Returns
    -------
    PosteriorPredictResult
        Object with .idata and .r attributes
    """
    import rpy2.robjects as ro
    get_brms()  # Ensure brms is loaded
    m = model.r

    data_r = py_to_r(newdata)
    kwargs = kwargs_r(kwargs)
    
    # Get R function explicitly
    r_posterior_predict = typing.cast(typing.Callable, ro.r('brms::posterior_predict'))
    
    # Call with proper arguments
    if newdata is not None:
        r = r_posterior_predict(m, newdata=data_r, **kwargs)
    else:
        r = r_posterior_predict(m, **kwargs)
    
    idata = brms_predict_to_idata(r, model.r, newdata=newdata)
    idata = typing.cast(IDPredict, idata)

    return PosteriorPredictResult(
        r=r, idata=idata
    )

def posterior_linpred(model: FitResult, newdata: typing.Optional[pd.DataFrame] = None, **kwargs) -> PosteriorLinpredResult:
    """
    Compute linear predictor of the model.
    
    Returns samples of the linear predictor (before applying the link function).
    Useful for understanding the model's predictions on the linear scale.

    Parameters
    ----------
    model : FitResult
        Fitted model from fit()
    newdata : pd.DataFrame, optional
        Data for predictions. If None, uses original data
    **kwargs : dict
        Additional arguments to brms::posterior_linpred():
        
        - transform : bool - Apply inverse link function (default False)
        - ndraws : int - Number of posterior draws
        - summary : bool - Return summary statistics
    
    Returns
    -------
    PosteriorLinpredResult
        Object with .idata (IDLinpred) and .r (R matrix) attributes
    
    See Also
    --------
    brms::posterior_linpred : R documentation
        https://paulbuerkner.com/brms/reference/posterior_linpred.brmsfit.html
    posterior_epred : Expected values on response scale
    
    Examples
    --------
    ```python
        from brmspy import brms
        
        epilepsy = brms.get_brms_data("epilepsy")
        model = brms.fit(
            "count ~ zAge + zBase * Trt + (1|patient)",
            data=epilepsy,
            family="poisson",
            chains=4
        )
        
        # Linear predictor (log scale for Poisson)
        linpred = brms.posterior_linpred(model)
        print(linpred.idata.predictions)
    ```
    """
    import rpy2.robjects as ro
    get_brms()  # Ensure brms is loaded
    m = model.r

    data_r = py_to_r(newdata)
    kwargs = kwargs_r(kwargs)
    
    # Get R function explicitly
    r_posterior_linpred = typing.cast(typing.Callable, ro.r('brms::posterior_linpred'))
    
    # Call with proper arguments
    if newdata is not None:
        r = r_posterior_linpred(m, newdata=data_r, **kwargs)
    else:
        r = r_posterior_linpred(m, **kwargs)
    
    idata = brms_linpred_to_idata(r, model.r, newdata=newdata)
    idata = typing.cast(IDLinpred, idata)

    return PosteriorLinpredResult(
        r=r, idata=idata
    )


def log_lik(model: FitResult, newdata: typing.Optional[pd.DataFrame] = None, **kwargs) -> LogLikResult:
    """
    Compute log-likelihood values.
    
    Returns log p(y|theta) for each observation. Essential for model
    comparison via LOO-CV and WAIC.

    Parameters
    ----------
    model : FitResult
        Fitted model from fit()
    newdata : pd.DataFrame, optional
        Data for predictions. If None, uses original data
    **kwargs : dict
        Additional arguments to brms::log_lik():
        
        - ndraws : int - Number of posterior draws
        - combine_chains : bool - Combine chains (default True)
    
    Returns
    -------
    LogLikResult
        Object with .idata (IDLogLik) and .r (R matrix) attributes
    
    See Also
    --------
    brms::log_lik : R documentation
        https://paulbuerkner.com/brms/reference/log_lik.brmsfit.html
    arviz.loo : Leave-One-Out Cross-Validation
    arviz.waic : Widely Applicable Information Criterion
    
    Examples
    --------
    Compute log-likelihood for model comparison:
    
    ```python
    from brmspy import brms
    import arviz as az
    
    epilepsy = brms.get_brms_data("epilepsy")
    model = brms.fit(
        "count ~ zAge + zBase * Trt + (1|patient)",
        data=epilepsy,
        family="poisson",
        chains=4
    )
    
    # LOO-CV for model comparison
    loo = az.loo(model.idata)
    print(loo)
    ```

    Compare multiple models:
    ```python
    model1 = brms.fit("count ~ zAge + (1|patient)", data=epilepsy, family="poisson", chains=4)
    model2 = brms.fit("count ~ zAge + zBase + (1|patient)", data=epilepsy, family="poisson", chains=4)
    
    comp = az.compare({'model1': model1.idata, 'model2': model2.idata})
    print(comp)
    ```
    """
    import rpy2.robjects as ro
    get_brms()  # Ensure brms is loaded
    m = model.r

    data_r = py_to_r(newdata)
    kwargs = kwargs_r(kwargs)
    
    # Get R function explicitly
    r_log_lik = typing.cast(typing.Callable, ro.r('brms::log_lik'))
    
    # Call with proper arguments
    if newdata is not None:
        r = r_log_lik(m, newdata=data_r, **kwargs)
    else:
        r = r_log_lik(m, **kwargs)
    
    idata = brms_log_lik_to_idata(r, model.r, newdata=newdata)
    idata = typing.cast(IDLogLik, idata)

    return LogLikResult(
        r=r, idata=idata
    )
