"""
Generic R function caller.

Use `call()` to invoke R functions by name (including brms functions) when there
is no dedicated wrapper in `brmspy.brms`.

Notes
-----
Executed inside the worker process that hosts the embedded R session.
"""

import re
from collections.abc import Callable
from typing import Any, cast

from brmspy.helpers._rpy2._conversion import kwargs_r, py_to_r, r_to_py


def sanitised_name(function: str) -> str:
    """
    Sanitize a function name for safe R execution.

    Parameters
    ----------
    function : str
        Function name (optionally namespaced, e.g. ``"brms::loo"``).

    Returns
    -------
    str
        A sanitized name where invalid characters are replaced with underscores,
        and where leading digits are avoided (except after a namespace).

    Examples
    --------
    >>> sanitised_name("my-function")
    'my_function'
    >>> sanitised_name("123func")
    '_123func'
    """
    # Replace invalid characters with underscores (except ::)
    sanitized = re.sub(r"[^a-zA-Z0-9_:.]", "_", function)

    # Ensure doesn't start with a number (unless it's after ::)
    parts = sanitized.split("::")
    parts = [("_" + part if part and part[0].isdigit() else part) for part in parts]

    return "::".join(parts)


def call(function: str, *args, **kwargs) -> Any:
    """
    Call an R function by name with brmspy type conversion.

    This is intended as an escape hatch for R/brms functionality that does not
    yet have a dedicated wrapper.

    Parameters
    ----------
    function : str
        Function name. If not namespaced, brmspy tries ``brms::<function>`` first,
        then falls back to evaluating the name directly (e.g. ``"stats::AIC"``).
    *args
        Positional arguments.
    **kwargs
        Keyword arguments.

    Returns
    -------
    Any
        Converted return value.

    Examples
    --------
    >>> from brmspy import brms
    >>> fit = brms.brm("y ~ x", data=df, chains=4)
    >>> aic = brms.call("stats::AIC", fit)
    """
    import rpy2.robjects as ro

    func_name = sanitised_name(function)
    args = [py_to_r(arg) for arg in args]
    kwargs = kwargs_r({**kwargs})
    try:
        r_fun = cast(
            Callable, ro.r(f"suppressWarnings(suppressMessages(brms::{func_name}))")
        )
    except Exception:
        r_fun = cast(Callable, ro.r(func_name))

    r_result = r_fun(*args, **kwargs)
    return r_to_py(r_result)
