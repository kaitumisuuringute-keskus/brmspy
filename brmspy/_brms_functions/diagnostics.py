"""
Diagnostic functions for brms models with ArviZ integration.

This module provides diagnostic functions for analyzing fitted brms models.
All fitted models return `arviz.InferenceData` objects by default through the
`.idata` attribute, enabling seamless integration with ArviZ's diagnostic toolkit.

ArviZ Integration
-----------------
brmspy models work directly with ArviZ functions without conversion:

- **Summary & Convergence**: `az.summary()`, `az.rhat()`, `az.ess()`
- **Visualization**: `az.plot_trace()`, `az.plot_posterior()`, `az.plot_pair()`
- **Model Comparison**: `az.loo()`, `az.waic()`, `az.compare()`
- **Predictive Checks**: `az.plot_ppc()`

**For multivariate models**, use the `var_name` parameter in ArviZ functions
to specify which response variable to analyze (e.g., `az.loo(model.idata, var_name="y1")`).

Quick Example
-------------
```python
import brmspy
import arviz as az

# Fit model
model = brmspy.fit("count ~ zAge + (1|patient)", data=data, family="poisson")

# Diagnostics
print(az.summary(model.idata))  # Parameter estimates with Rhat, ESS
az.plot_trace(model.idata)       # MCMC trace plots
az.plot_ppc(model.idata)         # Posterior predictive check

# Model comparison
loo = az.loo(model.idata)
print(loo)
```

See Also
--------
Diagnostics with ArviZ : Complete guide with examples
    [https://kaitumisuuringute-keskus.github.io/brmspy/api/diagnostics-arviz/](https://kaitumisuuringute-keskus.github.io/brmspy/api/diagnostics-arviz/)

Notes
-----
The InferenceData structure contains:

- **posterior**: All parameter samples with brms naming (e.g., `b_Intercept`, `sd_patient__Intercept`)
- **posterior_predictive**: Posterior predictive samples for each response
- **log_likelihood**: Pointwise log-likelihood for LOO/WAIC
- **observed_data**: Original response values
"""

from collections.abc import Callable, Iterable
from typing import cast

import numpy as np
import pandas as pd
import xarray as xr
from rpy2.rinterface import ListSexpVector

from brmspy.helpers._rpy2._robject_iter import iterate_robject_to_dataclass

from ..helpers._rpy2._conversion import kwargs_r, py_to_r, r_to_py
from ..types.brms_results import FitResult, SummaryResult


def summary(model: FitResult, **kwargs) -> SummaryResult:
    """
    Generate comprehensive summary statistics for a fitted brms model.

    Returns a `SummaryResult` dataclass containing model information,
    parameter estimates, and diagnostic information. The SummaryResult object provides
    pretty printing via `str()` or `print()` and structured access to all components.

    [BRMS documentation and parameters](https://paulbuerkner.com/brms/reference/summary.brmsfit.html)

    Parameters
    ----------
    model : FitResult
        Fitted model returned by `brmspy.brms.brm()`.
    **kwargs
        Additional arguments passed to brms::summary(), such as:
        - probs: Quantiles for credible intervals, e.g., `probs=(0.025, 0.975)`
        - robust: Use robust estimates (median, MAD), default False

    Returns
    -------
    SummaryResult
        A dataclass containing:

        - **formula** (str): Model formula as string
        - **data_name** (str): Name of the data object used
        - **group** (str): Grouping structure information
        - **nobs** (int): Number of observations
        - **ngrps** (Dict[str, int]): Number of groups per grouping variable
        - **autocor** (Optional[dict]): Autocorrelation structure if present
        - **prior** (pd.DataFrame): Prior specifications used
        - **algorithm** (str): Sampling algorithm (e.g., "sampling")
        - **sampler** (str): Sampler specification (e.g., "sample(hmc)")
        - **total_ndraws** (int): Total number of post-warmup draws
        - **chains** (float): Number of chains
        - **iter** (float): Iterations per chain
        - **warmup** (float): Warmup iterations per chain
        - **thin** (float): Thinning interval
        - **has_rhat** (bool): Whether Rhat diagnostics are reported
        - **fixed** (pd.DataFrame): Population-level (fixed) effects estimates
        - **spec_pars** (pd.DataFrame): Family-specific parameters (e.g., sigma)
        - **cor_pars** (pd.DataFrame): Correlation parameters if present
        - **random** (dict): Group-level (random) effects by grouping variable

    See Also
    --------
    brms::summary.brmsfit : [R documentation](https://paulbuerkner.com/brms/reference/summary.brmsfit.html)

    Examples
    --------
    Basic usage with pretty printing:

    ```python
    import brmspy

    model = brmspy.fit("y ~ x", data=data, chains=4)
    summary = brmspy.summary(model)

    # Pretty print full summary
    print(summary)
    ```

    Access specific components:

    ```python
    # Get population-level effects as DataFrame
    fixed_effects = summary.fixed
    print(fixed_effects)

    # Get family-specific parameters (e.g., sigma)
    spec_params = summary.spec_pars
    print(spec_params)

    # Access random effects (if present)
    random_effects = summary.random
    for group_name, group_df in random_effects.items():
        print(f"Random effects for {group_name}:")
        print(group_df)

    # Check model metadata
    print(f"Formula: {summary.formula}")
    print(f"Total draws: {summary.total_ndraws}")
    print(f"Rhat reported: {summary.has_rhat}")
    ```

    Custom credible intervals:

    ```python
    # Use 90% credible intervals instead of default 95%
    summary_90 = brmspy.summary(model, probs=(0.05, 0.95))
    print(summary_90.fixed)
    ```
    """

    import rpy2.robjects as ro

    kwargs = kwargs_r(kwargs)
    r_summary = cast(Callable, ro.r("summary"))
    summary_r = r_summary(model.r, **kwargs)

    _default_get_r = lambda param: f"function(x) x${param}"
    _get_methods_r: dict[str, Callable[[str], str]] = {
        # Extract a clean formula string: "y ~ x1 + x2 + ..."
        "formula": lambda param: (
            "function(x) { paste(deparse(x$formula$formula), collapse = ' ') }"
        ),
    }

    names = summary_r.names
    get = lambda param: r_to_py(
        cast(Callable, ro.r(_get_methods_r.get(param, _default_get_r)(param)))(
            summary_r
        )
    )
    out = iterate_robject_to_dataclass(
        names=names, get=get, target_dataclass=SummaryResult, r=summary_r
    )

    return cast(SummaryResult, out)


def fixef(
    object: FitResult | ListSexpVector,
    summary=True,
    robust=False,
    probs=(0.025, 0.975),
    pars=None,
    **kwargs,
) -> pd.DataFrame:
    """
    Extract population-level (fixed) effects estimates from a fitted brms model.

    Returns a pandas DataFrame containing estimates and uncertainty intervals for
    all population-level parameters (fixed effects). By default, returns summary
    statistics (mean, standard error, credible intervals). Can also return raw
    posterior samples when `summary=False`.

    [BRMS documentation](https://paulbuerkner.com/brms/reference/fixef.brmsfit.html)

    Parameters
    ----------
    object : FitResult or ListSexpVector
        Fitted model returned by `brmspy.brms.brm()` or an R brmsfit object.
    summary : bool, default=True
        If True, return summary statistics (mean/median, SE/MAD, credible intervals).
        If False, return matrix of posterior samples (iterations × parameters).
    robust : bool, default=False
        If True, use median and MAD instead of mean and SD for summary statistics.
        Only used when `summary=True`.
    probs : tuple of float, default=(0.025, 0.975)
        Quantiles for credible intervals, e.g., (0.025, 0.975) for 95% intervals.
        Only used when `summary=True`.
    pars : list of str, optional
        Specific parameter names to extract. If None, returns all fixed effects.
        Useful for subsetting when you only need specific coefficients.
    **kwargs
        Additional arguments passed to brms::fixef()

    Returns
    -------
    pd.DataFrame
        When `summary=True` (default):
            DataFrame with parameters as rows and columns for Estimate, Est.Error,
            Q2.5, Q97.5 (or other quantiles specified in `probs`), and optionally
            Rhat and Bulk_ESS/Tail_ESS diagnostics.

        When `summary=False`:
            DataFrame with posterior samples where rows are iterations and columns
            are parameters. Shape is (n_iterations × n_parameters).

    See Also
    --------
    brms::fixef.brmsfit : [R documentation](https://paulbuerkner.com/brms/reference/fixef.brmsfit.html)
    summary() : Full model summary with all parameter types

    Examples
    --------
    Basic usage with summary statistics:

    ```python
    import brmspy

    model = brmspy.fit("y ~ x1 + x2", data=data, chains=4)

    # Get fixed effects summary
    fixed_effects = brmspy.fixef(model)
    print(fixed_effects)
    #             Estimate  Est.Error      Q2.5     Q97.5
    # Intercept  10.234567   0.123456  9.992345  10.47689
    # x1          0.456789   0.098765  0.263456   0.65012
    # x2         -0.234567   0.087654 -0.406789  -0.06234
    ```

    Get specific parameters only:

    ```python
    # Extract only specific coefficients
    x1_x2_effects = brmspy.fixef(model, pars=["x1", "x2"])
    print(x1_x2_effects)
    ```

    Use robust estimates (median and MAD):

    ```python
    # Use median and MAD instead of mean and SD
    robust_effects = brmspy.fixef(model, robust=True)
    print(robust_effects)
    ```

    Custom credible intervals:

    ```python
    # Get 90% credible intervals
    effects_90 = brmspy.fixef(model, probs=(0.05, 0.95))
    print(effects_90)
    ```

    Get raw posterior samples:

    ```python
    # Get full posterior samples matrix
    samples = brmspy.fixef(model, summary=False)
    print(samples.shape)  # (n_iterations, n_parameters)

    # Can then compute custom statistics
    import numpy as np
    custom_quantile = np.percentile(samples["x1"], 90)
    ```
    """
    import rpy2.robjects as ro

    obj_r = py_to_r(object)
    kwargs = kwargs_r(
        {"summary": summary, "robust": robust, "probs": probs, "pars": pars, **kwargs}
    )
    r_fixef = cast(Callable, ro.r("brms::fixef"))
    r_df = r_fixef(obj_r, **kwargs)
    return cast(pd.DataFrame, r_to_py(r_df))


def ranef(
    object: FitResult | ListSexpVector,
    summary: bool = True,
    robust: bool = False,
    probs=(0.025, 0.975),
    pars=None,
    groups=None,
    **kwargs,
) -> dict[str, xr.DataArray]:
    """
    Extract group-level (random) effects as xarray DataArrays.

    This is a wrapper around ``brms::ranef()``. For ``summary=True`` (default),
    each grouping factor is returned as a 3D array with dimensions
    ``("group", "stat", "coef")``. For ``summary=False``, each factor is
    returned as ``("draw", "group", "coef")`` with one slice per posterior draw.

    Parameters
    ----------
    object : FitResult or rpy2.robjects.ListVector
        Fitted model returned by :func:`brmspy.brms.fit` or an R ``brmsfit``
        object / summary list.
    summary : bool, default True
        If True, return posterior summaries for the group-level effects
        (means, errors, intervals). If False, return per-draw random effects.
    robust : bool, default False
        If True, use robust summaries (median and MAD) instead of mean and SD.
        Passed through to ``brms::ranef()`` when ``summary=True``.
    probs : tuple of float, default (0.025, 0.975)
        Central posterior interval probabilities, as in ``brms::ranef()``.
        Only used when ``summary=True``.
    pars : str or sequence of str, optional
        Subset of group-level parameters to include. Passed to ``brms::ranef()``.
    groups : str or sequence of str, optional
        Subset of grouping factors to include. Passed to ``brms::ranef()``.
    **kwargs
        Additional keyword arguments forwarded to ``brms::ranef()``.

    Returns
    -------
    dict[str, xarray.DataArray]
        Mapping from grouping-factor name (e.g. ``"patient"``) to a
        ``DataArray``:

        * ``summary=True``: dims ``("group", "stat", "coef")``,
          with ``stat`` typically containing
          ``["Estimate", "Est.Error", "Q2.5", "Q97.5"]``.
        * ``summary=False``: dims ``("draw", "group", "coef")``,
          where ``draw`` indexes posterior samples.

    Examples
    --------
    Compute summary random effects and inspect all coefficients for a single
    group level:

    ```python
    from brmspy import brms
    from brmspy.brms import ranef

    fit = brms.fit("count ~ zAge + zBase * Trt + (1 + zBase + Trt | patient)",
                   data=data, family="poisson")

    re = ranef(fit)  # summary=True by default
    patient_re = re["patient"].sel(group="1", stat="Estimate")
    ```

    Extract per-draw random effects for downstream MCMC analysis:

    ```python
    re_draws = ranef(fit, summary=False)
    patient_draws = re_draws["patient"]       # dims: ("draw", "group", "coef")
    first_draw = patient_draws.sel(draw=0)
    ```
    """
    import rpy2.robjects as ro

    obj_r = py_to_r(object)
    kwargs = kwargs_r(
        {"summary": summary, "robust": robust, "probs": probs, "pars": pars, **kwargs}
    )

    r_ranef = cast(Callable, ro.r("brms::ranef"))
    r_list = r_ranef(obj_r, **kwargs)

    out: dict[str, xr.DataArray] = {}

    for name in r_list.names:
        # R 3D array for this grouping factor
        r_arr = cast(Callable, ro.r(f"function(x) x${name}"))(r_list)
        dims = list(r_arr.do_slot("dim"))  # length-3

        # dimnames is a list of length 3, some entries may be NULL
        dimnames_r = r_arr.do_slot("dimnames")
        dimnames: list[list[str] | None] = []
        for dn in dimnames_r:
            if dn == ro.NULL:
                dimnames.append(None)
            else:
                dimnames.append(list(cast(Iterable, r_to_py(dn))))

        p_arr = np.asarray(r_arr).reshape(dims)

        if summary:
            # brms: 1=group levels, 2=stats, 3=coefs
            groups_dn, stats_dn, coefs_dn = dimnames

            da = xr.DataArray(
                p_arr,
                dims=("group", "stat", "coef"),
                coords={
                    "group": groups_dn,
                    "stat": stats_dn,
                    "coef": coefs_dn,
                },
            )
        else:
            # brms: 1=draws, 2=group levels, 3=coefs
            draws_dn, groups_dn, coefs_dn = dimnames
            n_draws = dims[0]
            if draws_dn is None:
                # brms does not name draws, so create a simple index
                draws_dn = list(range(n_draws))

            da = xr.DataArray(
                p_arr,
                dims=("draw", "group", "coef"),
                coords={
                    "draw": draws_dn,
                    "group": groups_dn,
                    "coef": coefs_dn,
                },
            )

        out[name] = da
    return out


def posterior_summary(
    object: FitResult | ListSexpVector,
    variable=None,
    probs=(0.025, 0.975),
    robust=False,
    **kwargs,
) -> pd.DataFrame:
    """
    Extract posterior summary statistics for all or selected model parameters.

    Provides a DataFrame with estimates, standard errors, and credible intervals
    for all parameters in a brms model, including fixed effects, random effects,
    and auxiliary parameters. More comprehensive than `fixef()` or `ranef()` as it covers all
    parameter types.

    [BRMS documentation](https://paulbuerkner.com/brms/reference/posterior_summary.brmsfit.html)

    Parameters
    ----------
    object : FitResult or ListSexpVector
        Fitted model returned by `brmspy.brms.brm()` or an R brmsfit object.
    variable : str or list of str, optional
        Specific variable name(s) to extract. If None, returns all parameters.
        Supports regex patterns for flexible selection.
    probs : tuple of float, default=(0.025, 0.975)
        Quantiles for credible intervals, e.g., (0.025, 0.975) for 95% intervals.
    robust : bool, default=False
        If True, use median and MAD instead of mean and SD for summary statistics.
    **kwargs
        Additional arguments passed to brms::posterior_summary()

    Returns
    -------
    pd.DataFrame
        DataFrame with parameters as rows and columns for Estimate, Est.Error,
        and quantiles (e.g., Q2.5, Q97.5). Includes all model parameters:
        population-level effects, group-level effects, and auxiliary parameters.

    See Also
    --------
    brms::posterior_summary : R documentation
        https://paulbuerkner.com/brms/reference/posterior_summary.brmsfit.html
    fixef() : Extract only population-level effects
    ranef() : Extract only group-level effects

    Examples
    --------
    Get summary for all parameters:

    ```python
    import brmspy

    model = brmspy.fit("y ~ x1 + (1|group)", data=data, chains=4)

    # Get all parameter summaries
    all_params = brmspy.posterior_summary(model)
    print(all_params)
    ```

    Extract specific parameters:

    ```python
    # Get summary for specific parameters
    intercept = brmspy.posterior_summary(model, variable="b_Intercept")
    print(intercept)

    # Multiple parameters
    fixed_only = brmspy.posterior_summary(model, variable=["b_Intercept", "b_x1"])
    print(fixed_only)
    ```

    Custom credible intervals with robust estimates:

    ```python
    # 90% intervals with median/MAD
    robust_summary = brmspy.posterior_summary(
        model,
        probs=(0.05, 0.95),
        robust=True
    )
    print(robust_summary)
    ```
    """
    import rpy2.robjects as ro

    obj_r = py_to_r(object)
    kwargs = kwargs_r(
        {"variable": variable, "probs": probs, "robust": robust, **kwargs}
    )

    r_fun = cast(Callable, ro.r("brms::posterior_summary"))
    r_df = r_fun(obj_r, **kwargs)
    return cast(pd.DataFrame, r_to_py(r_df))


def prior_summary(
    object: FitResult | ListSexpVector, all=True, **kwargs
) -> pd.DataFrame:
    """
    Extract prior specifications used in a fitted brms model.

    Returns a DataFrame containing all prior distributions that were used
    (either explicitly set or defaults) when fitting the model. Useful for
    documenting model specifications and understanding which priors were applied.

    [BRMS documentation](https://paulbuerkner.com/brms/reference/prior_summary.brmsfit.html)

    Parameters
    ----------
    object : FitResult or ListVector
        Fitted model returned by `brmspy.brms.brm()` or an R brmsfit object.
    all : bool, default=True
        If True, return all priors including default priors.
        If False, return only explicitly set priors.
    **kwargs
        Additional arguments passed to brms::prior_summary()

    Returns
    -------
    pd.DataFrame
        DataFrame with columns describing prior specifications:
        - prior: Prior distribution formula
        - class: Parameter class (b, sd, Intercept, etc.)
        - coef: Specific coefficient (if applicable)
        - group: Grouping factor (if applicable)
        - resp: Response variable (for multivariate models)
        - dpar: Distributional parameter (if applicable)
        - nlpar: Non-linear parameter (if applicable)
        - lb/ub: Bounds for truncated priors
        - source: Origin of prior (default, user, etc.)

    See Also
    --------
    brms::prior_summary : R documentation
        https://paulbuerkner.com/brms/reference/prior_summary.brmsfit.html
    get_prior() : Get prior structure before fitting
    default_prior() : Get default priors for a model

    Examples
    --------
    Get all priors used in a model:

    ```python
    import brmspy

    model = brmspy.fit(
        "y ~ x1 + (1|group)",
        data=data,
        priors=[brmspy.prior("normal(0, 1)", "b")],
        chains=4
    )

    # Get all priors (including defaults)
    priors = brmspy.prior_summary(model)
    print(priors)
    ```

    Get only explicitly set priors:

    ```python
    # Get only user-specified priors
    user_priors = brmspy.prior_summary(model, all=False)
    print(user_priors)
    ```

    Compare with what would be used before fitting:

    ```python
    # Before fitting - check default priors
    default_priors = brmspy.get_prior("y ~ x1", data=data)

    # After fitting - see what was actually used
    used_priors = brmspy.prior_summary(model)
    ```
    """
    import rpy2.robjects as ro

    obj_r = py_to_r(object)
    kwargs = kwargs_r({"all": all, **kwargs})

    r_fun = cast(Callable, ro.r("brms::prior_summary"))
    r_df = r_fun(obj_r, **kwargs)
    return cast(pd.DataFrame, r_to_py(r_df))


def validate_newdata(
    newdata: pd.DataFrame,
    object: ListSexpVector | FitResult,
    re_formula: str | None = None,
    allow_new_levels: bool = False,
    newdata2: pd.DataFrame | None = None,
    resp=None,
    check_response=True,
    incl_autocor=True,
    group_vars=None,
    req_vars=None,
    **kwargs,
) -> pd.DataFrame:
    """
    Validate new data for predictions from a fitted brms model.

    Ensures that new data contains all required variables and has the correct
    structure for making predictions. Checks variable types, factor levels,
    grouping variables, and autocorrelation structures. This function is primarily
    used internally by prediction methods but can be called directly for debugging
    or validation purposes.

    [BRMS documentation](https://paulbuerkner.com/brms/reference/validate_newdata.html)

    Parameters
    ----------
    newdata : pd.DataFrame
        DataFrame containing new data to be validated against the model.
        Must include all predictor variables used in the model formula.
    object : FitResult or ListSexpVector
        Fitted model returned by `brmspy.brms.brm()` or an R brmsfit object.
    re_formula : str, optional
        Formula string specifying group-level effects to include in validation.
        If None (default), include all group-level effects.
        If NA, include no group-level effects.
    allow_new_levels : bool, default=False
        Whether to allow new levels of grouping variables not present in
        the original training data. If False, raises an error for new levels.
    newdata2 : pd.DataFrame, optional
        Additional data that cannot be passed via `newdata`, such as objects
        used in autocorrelation structures or stanvars.
    resp : str or list of str, optional
        Names of response variables to validate. If specified, validation
        is performed only for the specified responses (relevant for multivariate models).
    check_response : bool, default=True
        Whether to check if response variables are present in newdata.
        Set to False when making predictions where response is not needed.
    incl_autocor : bool, default=True
        Whether to include autocorrelation structures originally specified
        in the model. If True, validates autocorrelation-related variables.
    group_vars : list of str, optional
        Names of specific grouping variables to validate. If None (default),
        validates all grouping variables present in the model.
    req_vars : list of str, optional
        Names of specific variables required in newdata. If None (default),
        all variables from the original training data are required (unless
        excluded by other parameters).
    **kwargs
        Additional arguments passed to brms::validate_newdata()

    Returns
    -------
    pd.DataFrame
        Validated DataFrame based on newdata, potentially with added or
        modified columns to ensure compatibility with the model.

    Raises
    ------
    ValueError
        If newdata is missing required variables
    ValueError
        If factor levels in newdata don't match those in training data
        (when allow_new_levels=False)
    ValueError
        If grouping variables have invalid structure

    See Also
    --------
    brms::validate_newdata : R documentation
        https://paulbuerkner.com/brms/reference/validate_newdata.html
    posterior_predict() : Uses validate_newdata internally
    posterior_epred() : Uses validate_newdata internally

    Examples
    --------
    Basic validation for prediction data:

    ```python
    import brmspy
    import pandas as pd

    # Fit model
    model = brmspy.fit("y ~ x1 + x2", data=train_data, chains=4)

    # Prepare new data
    new_data = pd.DataFrame({
        'x1': [1.0, 2.0, 3.0],
        'x2': [0.5, 1.0, 1.5]
    })

    # Validate before prediction
    validated_data = brmspy.validate_newdata(new_data, model)
    print(validated_data)
    ```

    Validate with group-level effects:

    ```python
    # Model with random effects
    model = brmspy.fit("y ~ x + (1|group)", data=train_data, chains=4)

    # New data with grouping variable
    new_data = pd.DataFrame({
        'x': [1.0, 2.0],
        'group': ['A', 'B']  # Must match training data groups
    })

    # Validate - will error if groups A or B weren't in training
    validated_data = brmspy.validate_newdata(
        new_data,
        model,
        allow_new_levels=False
    )
    ```

    Allow new levels for population-level predictions:

    ```python
    # Allow new group levels (makes population-level predictions only)
    new_data_with_new_groups = pd.DataFrame({
        'x': [3.0, 4.0],
        'group': ['C', 'D']  # New groups not in training
    })

    validated_data = brmspy.validate_newdata(
        new_data_with_new_groups,
        model,
        allow_new_levels=True
    )
    ```

    Skip response variable checking:

    ```python
    # When making predictions, response not needed
    new_data = pd.DataFrame({'x1': [1.0, 2.0]})

    validated_data = brmspy.validate_newdata(
        new_data,
        model,
        check_response=False
    )
    ```
    """
    import rpy2.robjects as ro

    r_validate_newdata = cast(Callable, ro.r("brms::validate_newdata"))
    kwargs = kwargs_r(
        {
            "newdata": newdata,
            "object": object,
            "re_formula": re_formula,
            "allow_new_levels": allow_new_levels,
            "newdata2": newdata2,
            "resp": resp,
            "check_response": check_response,
            "incl_autocor": incl_autocor,
            "group_vars": group_vars,
            "req_vars": req_vars,
            **kwargs,
        }
    )
    res_r = r_validate_newdata(**kwargs)
    return cast(pd.DataFrame, r_to_py(res_r))
