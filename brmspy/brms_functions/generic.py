from typing import Any, Callable, cast
import re

from brmspy.helpers.conversion import kwargs_r, py_to_r, r_to_py
import rpy2.robjects as ro


def sanitised_name(function: str) -> str:
    """
    Sanitize a function name for safe R execution.
    
    Converts Python-style function names to valid R identifiers by:
    - Replacing invalid characters with underscores
    - Ensuring the name doesn't start with a number
    - Preserving namespace separators (::)
    
    Parameters
    ----------
    function : str
        Function name to sanitize
    
    Returns
    -------
    str
        Sanitized function name safe for R execution
    
    Examples
    --------
    ```python
    from brmspy.brms_functions.generic import sanitised_name
    
    # Basic sanitization
    print(sanitised_name("my-function"))  # "my_function"
    print(sanitised_name("123func"))       # "_123func"
    
    # Preserves namespace
    print(sanitised_name("brms::loo"))     # "brms::loo"
    ```
    """
    # Replace invalid characters with underscores (except ::)
    sanitized = re.sub(r'[^a-zA-Z0-9_:.]', '_', function)
    
    # Ensure doesn't start with a number (unless it's after ::)
    parts = sanitized.split('::')
    parts = [('_' + part if part and part[0].isdigit() else part) for part in parts]
    
    return '::'.join(parts)


def call(
  function: str,
  *args,
  **kwargs
) -> Any:
    """
    Call any brms or R function by name with automatic type conversion.
    
    Generic wrapper for calling brms functions that don't have dedicated Python
    wrappers. Automatically converts Python arguments to R objects and R results
    back to Python. Tries `brms::function_name` first, then falls back to base R.
    
    This function is useful for:
    - Accessing newer brms functions not yet wrapped in brmspy
    - Calling brms utility functions without writing custom wrappers
    - Quick exploration of brms functionality from Python
    
    Parameters
    ----------
    function : str
        Name of the R function to call. Will be prefixed with 'brms::' if possible.
        Can also include namespace (e.g., "stats::predict").
    *args
        Positional arguments passed to the R function. Automatically converted
        from Python to R types (FitResult → brmsfit, DataFrame → data.frame, etc.).
    **kwargs
        Keyword arguments passed to the R function. Python parameter names are
        automatically converted to R conventions.
    
    Returns
    -------
    Any
        Result from R function, automatically converted to appropriate Python type
        (R data.frame → pandas DataFrame, R vector → numpy array, etc.).
    
    See Also
    --------
    [`py_to_r`](brmspy/helpers/conversion.py:1) : Python to R type conversion
    [`r_to_py`](brmspy/helpers/conversion.py:1) : R to Python type conversion
    
    Examples
    --------
    Call brms functions not yet wrapped:
    
    ```python
    from brmspy import brms
    from brmspy.brms_functions.generic import call
    
    model = brms.fit("y ~ x", data=data, chains=4)
    
    # Call brms::neff_ratio (not yet wrapped)
    neff = call("neff_ratio", model)
    print(neff)
    
    # Call brms::rhat (not yet wrapped)
    rhat = call("rhat", model)
    print(rhat)
    ```
    
    Call with keyword arguments:
    
    ```python
    # Call brms::hypothesis for testing hypotheses
    hypothesis_result = call(
        "hypothesis",
        model,
        hypothesis="b_x1 > 0",
        alpha=0.05
    )
    print(hypothesis_result)
    ```
    
    Access functions from other R packages:
    
    ```python
    # Call functions with namespace
    result = call("stats::AIC", model)
    print(result)
    ```
    """
    
    func_name = sanitised_name(function)
    args = [py_to_r(arg) for arg in args]
    kwargs = kwargs_r({
        **kwargs
    })
    try:
        r_fun = cast(Callable, ro.r(f'brms::{func_name}'))
    except Exception as e:
        r_fun = cast(Callable, ro.r(func_name))

    r_result = r_fun(*args, **kwargs)
    return r_to_py(r_result)

