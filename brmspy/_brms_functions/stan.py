"""
Stan code helpers.

This module exposes wrappers for generating Stan code and data from brms models
without running sampling.

Notes
-----
Executed inside the worker process that hosts the embedded R session.
"""

import typing

import pandas as pd

from brmspy.types.formula_dsl import FormulaConstruct, FormulaPart

from ..helpers._rpy2._conversion import _coerce_stan_types, py_to_r, r_to_py
from ..helpers._rpy2._priors import _build_priors
from ..types.brms_results import PriorSpec
from .formula import _execute_formula, bf

_formula_fn = bf


def make_stancode(
    formula: FormulaConstruct | str,
    data: pd.DataFrame,
    priors: typing.Sequence[PriorSpec] | None = None,
    family: str = "poisson",
    sample_prior: str = "no",
    formula_args: dict | None = None,
) -> str:
    """
    Generate Stan code using R ``brms::make_stancode()``.

    Useful for inspecting the generated Stan model before fitting.

    Parameters
    ----------
    formula : str or FormulaConstruct
        Model formula.
    data : pandas.DataFrame
        Model data.
    priors : Sequence[PriorSpec] or None, default=None
        Optional prior specifications created via `brmspy.brms.prior()`.
    family : str, default="poisson"
        Distribution family (e.g. ``"gaussian"``, ``"poisson"``).
    sample_prior : str, default="no"
        Prior sampling mode passed to brms (``"no"``, ``"yes"``, ``"only"``).
    formula_args : dict or None, default=None
        Reserved for future use. Currently ignored.

    Returns
    -------
    str
        Complete Stan program as a string.

    See Also
    --------
    brms::make_stancode : [R documentation](https://paulbuerkner.com/brms/reference/make_stancode.html)

    Examples
    --------
    ```python
    from brmspy import brms

    epilepsy = brms.get_brms_data("epilepsy")
    code = brms.make_stancode(
        "count ~ zAge + zBase * Trt + (1|patient)",
        data=epilepsy,
        family="poisson",
    )

    assert isinstance(code, str)
    ```
    """
    import rpy2.robjects as ro

    fun_make_stancode = typing.cast(typing.Callable, ro.r("brms::make_stancode"))

    data_r = py_to_r(data)
    priors_r = _build_priors(priors)
    if isinstance(formula, FormulaConstruct):
        formula_obj = _execute_formula(formula)
    else:
        if formula_args is None:
            formula_args = {}
        formula = FormulaConstruct._formula_parse(formula)
        formula_obj = _execute_formula(formula)

    if len(priors_r) > 0:
        return fun_make_stancode(
            formula=formula_obj,
            data=data_r,
            prior=priors_r,
            family=family,
            sample_prior=sample_prior,
        )[0]
    else:
        return fun_make_stancode(
            formula=formula_obj, data=data_r, family=family, sample_prior=sample_prior
        )[0]


def make_standata(
    formula: FormulaConstruct | str,
    data: pd.DataFrame,
    priors: typing.Sequence[PriorSpec] | None = None,
    family: str = "gaussian",
    sample_prior: str = "no",
    formula_args: dict | None = None,
    coerce_types: bool = True,
) -> dict:
    """
    Extract the Stan data dictionary using R ``brms::make_standata()``.

    Returns the exact data list that brms would pass to Stan, converted to a
    plain Python dict.  Optionally the dict is type-coerced so that Stan
    ``int`` variables carry Python ``int`` / ``numpy.int64`` values (required
    by nutpie and CmdStanPy).

    Parameters
    ----------
    formula : str or FormulaConstruct
        Model formula.
    data : pandas.DataFrame
        Model data.
    priors : Sequence[PriorSpec] or None, default=None
        Optional prior specifications created via `brmspy.brms.prior()`.
    family : str, default="gaussian"
        Distribution family (e.g. ``"gaussian"``, ``"poisson"``).
    sample_prior : str, default="no"
        Prior sampling mode passed to brms (``"no"``, ``"yes"``, ``"only"``).
    formula_args : dict or None, default=None
        Reserved for future use. Currently ignored.
    coerce_types : bool, default=True
        When ``True``, integer variables declared in the Stan data block are
        cast to ``int`` / ``numpy.int64`` so that nutpie / CmdStanPy accept
        them without complaint.  Requires ``make_stancode()`` to be called
        internally as well; set to ``False`` to skip this step if you already
        handle type coercion yourself.

    Returns
    -------
    dict
        Stan data dictionary with Python-native values (numpy arrays, scalars).

    See Also
    --------
    make_stancode : Generate the accompanying Stan program string.
    brms::make_standata : [R documentation](https://paulbuerkner.com/brms/reference/make_standata.html)

    Examples
    --------
    ```python
    from brmspy import brms

    epilepsy = brms.get_brms_data("epilepsy")
    stan_data = brms.make_standata(
        "count ~ zAge + zBase * Trt + (1|patient)",
        data=epilepsy,
        family="poisson",
    )

    assert isinstance(stan_data, dict)
    assert "N" in stan_data
    ```
    """
    import rpy2.robjects as ro

    fun_make_standata = typing.cast(typing.Callable, ro.r("brms::make_standata"))

    data_r = py_to_r(data)
    priors_r = _build_priors(priors)

    if isinstance(formula, FormulaConstruct):
        formula_obj = _execute_formula(formula)
    else:
        if formula_args is None:
            formula_args = {}
        formula = FormulaConstruct._formula_parse(formula)
        formula_obj = _execute_formula(formula)

    call_kwargs: dict = {
        "formula": formula_obj,
        "data": data_r,
        "family": family,
        "sample_prior": sample_prior,
    }
    if len(priors_r) > 0:
        call_kwargs["prior"] = priors_r

    stan_data_r = fun_make_standata(**call_kwargs)

    # Convert the R list to a plain Python dict
    stan_data: dict = typing.cast(dict, r_to_py(stan_data_r))

    # r_to_py may return ShmArray objects (np.ndarray subclasses backed by
    # shared memory) when called inside the worker process. nutpie's
    # .with_data() tries to JSON-serialise the values and doesn't know about
    # ShmArray, so we materialise every array value into a plain np.ndarray.
    import numpy as np

    stan_data = {
        k: np.array(v) if isinstance(v, np.ndarray) else v for k, v in stan_data.items()
    }

    if coerce_types:
        stan_code = make_stancode(
            formula=formula,
            data=data,
            priors=priors,
            family=family,
            sample_prior=sample_prior,
        )
        stan_data = _coerce_stan_types(stan_code, stan_data)

    return stan_data
