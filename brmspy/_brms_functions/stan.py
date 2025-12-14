"""
Stan code helpers.

This module exposes wrappers for generating Stan code from brms models without
running sampling.

Notes
-----
Executed inside the worker process that hosts the embedded R session.
"""

import typing

import pandas as pd

from brmspy.types.formula_dsl import FormulaConstruct, FormulaPart

from ..helpers._rpy2._conversion import py_to_r
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
