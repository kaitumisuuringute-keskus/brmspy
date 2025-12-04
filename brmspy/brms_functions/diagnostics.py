from dataclasses import dataclass
from types import UnionType
from typing import Any, Callable, Dict, Optional, TypedDict, Union, cast, get_origin
import typing
import pandas as pd
from rpy2.robjects import default_converter, pandas2ri
import pandas as pd

from brmspy.helpers.log import log_warning
from brmspy.helpers.robject_iter import iterate_robject_to_dataclass
from ..helpers.conversion import (
    kwargs_r,
    py_to_r,
    r_to_py
)
from ..types import (
    FitResult,
    SummaryResult
)
import rpy2.robjects as ro




def summary(model: FitResult, **kwargs) -> SummaryResult:
    """
    Generate comprehensive summary statistics for fitted brms model.
    
    Returns a `SummaryResult` dataclass containing model information,
    parameter estimates, and diagnostic information. The SummaryResult object provides
    pretty printing via `str()` or `print()` and structured access to all components.
    
    [BRMS documentation and parameters](https://paulbuerkner.com/brms/reference/summary.brmsfit.html)
    
    Parameters
    ----------
    model : FitResult
        Fitted model from [`fit()`](brmspy/brms_functions/brm.py:1)
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
    brms::summary.brmsfit : R documentation
        https://paulbuerkner.com/brms/reference/summary.brmsfit.html
    
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
    
    kwargs = kwargs_r(kwargs)
    r_summary = cast(Callable, ro.r('summary'))
    summary_r = r_summary(model.r, **kwargs)


    _default_get_r = lambda param: f'function(x) x${param}'
    _get_methods_r: Dict[str, Callable[[str], str]] = {
        # Extract a clean formula string: "y ~ x1 + x2 + ..."
        "formula": lambda param: (
            "function(x) { paste(deparse(x$formula$formula), collapse = ' ') }"
        ),
    }

    names = summary_r.names
    get = lambda param: r_to_py(cast(Callable, ro.r(_get_methods_r.get(param, _default_get_r)(param)))(summary_r))
    out = iterate_robject_to_dataclass(names=names, get=get, target_dataclass=SummaryResult)

    return cast(SummaryResult, out)


def fixef(
  object: Union[FitResult, ro.ListVector],
  summary = True,
  robust = False,
  probs = (0.025, 0.975),
  pars = None,
  **kwargs
) -> pd.DataFrame:
    obj_r = py_to_r(object)
    kwargs = kwargs_r(kwargs)

    return pd.DataFrame({})