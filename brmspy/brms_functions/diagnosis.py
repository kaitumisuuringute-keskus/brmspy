import typing
import pandas as pd
from rpy2.robjects import default_converter, pandas2ri

from ..helpers.conversion import (
    kwargs_r
)
from ..types import (
    FitResult
)
import rpy2.robjects as ro


def summary(model: FitResult, **kwargs) -> pd.DataFrame:
    """
    Generate summary statistics for fitted model.
    
    Calls R's summary() function on brmsfit object and converts to pandas DataFrame.

    [BRMS documentation and parameters](https://paulbuerkner.com/brms/reference/summary.brmsfit.html)
    
    Parameters
    ----------
    model : FitResult
        Fitted model from fit()
    **kwargs
        Additional arguments to summary(), e.g., probs=c(0.025, 0.975)
    
    Returns
    -------
    pd.DataFrame
        Summary statistics with columns for Estimate, Est.Error, and credible intervals
    
    See Also
    --------
    brms::summary.brmsfit : R documentation
        https://paulbuerkner.com/brms/reference/summary.brmsfit.html
    
    Examples
    --------
    ```python
    from brmspy import brms
    
    model = brms.fit("y ~ x", data=data, chains=4)
    summary_df = brms.summary(model)
    print(summary_df)
    ```
    """
    from rpy2.robjects.conversion import localconverter
    
    kwargs = kwargs_r(kwargs)

    # Get R summary function
    r_summary = typing.cast(typing.Callable, ro.r('summary'))
    
    # Call summary on brmsfit object
    summary_r = r_summary(model.r, **kwargs)
    
    # Extract the fixed effects table (summary$fixed)
    # brms summary returns a list with $fixed, $random, $spec_pars, etc.
    try:
        fixed_table = typing.cast(typing.Callable, ro.r('function(x) as.data.frame(x$fixed)'))(summary_r)
        
        # Convert to pandas
        with localconverter(default_converter + pandas2ri.converter):
            summary_df = pandas2ri.rpy2py(fixed_table)
        
        return summary_df
    except Exception:
        # Fallback: just convert the whole summary to string
        summary_str = str(summary_r)
        return pd.DataFrame({'summary': [summary_str]})
    
