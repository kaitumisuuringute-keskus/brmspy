"""
Formula helpers and DSL.

This module provides a small Pythonic DSL for composing brms formulas. The public
functions (`bf`, `lf`, `nlf`, `acformula`, `set_rescor`, `set_mecor`, `set_nl`)
build a `FormulaConstruct` that can be passed to `brmspy.brms.brm()` or combined
using the ``+`` operator.

Notes
-----
- The returned objects are lightweight formula specifications; the actual R brms
  formula object is built in the worker when fitting / generating Stan code.
- This module is part of the public API documented under docs/api/brms_functions/formula.md.
"""

from collections.abc import Callable
from typing import cast, get_args

from rpy2.rinterface_lib.sexp import Sexp

from brmspy.helpers.log import log
from brmspy.types.session import SexpWrapper

from ..helpers._rpy2._conversion import kwargs_r, py_to_r
from ..types.brms_results import ProxyListSexpVector
from ..types.formula_dsl import (
    _FORMULA_FUNCTION_WHITELIST,
    FormulaConstruct,
    FormulaPart,
)


def bf(*formulas: str, **formula_args) -> FormulaConstruct:
    """
    Build a brms model formula.

    This is the primary entrypoint for specifying the mean model and can be
    combined with other formula parts (e.g. `lf`, `nlf`, `acformula`) using ``+``.

    Parameters
    ----------
    *formulas : str
        One or more brms formula strings (e.g. ``"y ~ x + (1|group)"``). Multiple
        formulas are commonly used for multivariate models.
    **formula_args
        Keyword arguments forwarded to R ``brms::brmsformula()`` (for example
        ``decomp="QR"``, ``center=True``, ``sparse=True``, ``nl=True``, ``loop=True``).

    Returns
    -------
    FormulaConstruct
        A composable formula specification.

    See Also
    --------
    brms::brmsformula : [R documentation](https://paulbuerkner.com/brms/reference/brmsformula.html)

    Examples
    --------
    Basic formula:

    ```python
    from brmspy.brms import bf

    f = bf("y ~ x1 + x2 + (1|group)")
    ```

    QR decomposition (often helps with collinearity):

    ```python
    from brmspy.brms import bf

    f = bf("reaction ~ days + (days|subject)", decomp="QR")
    ```

    Multivariate formula + residual correlation:

    ```python
    from brmspy.brms import bf, set_rescor

    f = bf("mvbind(y1, y2) ~ x") + set_rescor(True)
    ```
    """
    part = FormulaPart(_fun="bf", _args=list(formulas), _kwargs=formula_args)
    return FormulaConstruct._formula_parse(part)


def lf(
    *formulas: str | FormulaConstruct | FormulaPart | ProxyListSexpVector,
    flist=None,
    dpar: str | None = None,
    resp: str | None = None,
    center: bool | None = None,
    cmc: bool | None = None,
    sparse: bool | None = None,
    decomp: str | None = None,
) -> FormulaConstruct:
    """
    Add linear formulas for distributional / non-linear parameters.

    This wraps R ``brms::lf()`` and is typically used to model distributional
    parameters such as ``sigma`` (heteroskedasticity) or to specify predictors
    for non-linear parameters.

    Parameters
    ----------
    *formulas
        One or more formulas such as ``"sigma ~ x"``.
    flist
        Optional list of formulas (advanced; mirrors brms).
    dpar : str or None, default=None
        Distributional parameter name (e.g. ``"sigma"``, ``"phi"``).
    resp : str or None, default=None
        Response name for multivariate models.
    center, cmc, sparse, decomp
        Forwarded to R ``brms::lf()``.

    Returns
    -------
    FormulaConstruct
        A composable formula specification that can be combined using ``+``.

    See Also
    --------
    brms::lf : [R documentation](https://paulbuerkner.com/brms/reference/lf.html)

    Examples
    --------
    Model mean + sigma:

    ```python
    from brmspy.brms import bf, lf

    f = bf("y ~ x") + lf("sigma ~ x", dpar="sigma")
    ```
    """
    formula_args = {
        "flist": flist,
        "dpar": dpar,
        "resp": resp,
        "center": center,
        "cmc": cmc,
        "sparse": sparse,
        "decomp": decomp,
    }
    result = FormulaConstruct._formula_parse(
        FormulaPart("lf", list(formulas), formula_args)
    )
    return result


def nlf(
    *formulas: str | FormulaConstruct | FormulaPart | ProxyListSexpVector,
    flist=None,
    dpar: str | None = None,
    resp: str | None = None,
    loop: bool | None = None,
) -> FormulaConstruct:
    """
    Add non-linear formulas.

    Wraps R ``brms::nlf()``. This is used together with `set_nl()` and parameter
    definitions in `lf()` to specify non-linear models.

    Parameters
    ----------
    *formulas
        One or more non-linear formulas (e.g. ``"y ~ a * exp(b * x)"``).
    flist
        Optional list of formulas (advanced; mirrors brms).
    dpar : str or None, default=None
        Distributional parameter name (optional).
    resp : str or None, default=None
        Response name for multivariate models.
    loop : bool or None, default=None
        Forwarded to R ``brms::nlf(loop=...)``.

    Returns
    -------
    FormulaConstruct
        A composable formula specification.

    See Also
    --------
    brms::nlf : [R documentation](https://paulbuerkner.com/brms/reference/nlf.html)

    Examples
    --------
    ```python
    from brmspy.brms import bf, nlf, set_nl

    f = bf("y ~ 1") + nlf("y ~ a * exp(b * x)") + set_nl()
    ```
    """
    formula_args = {
        "flist": flist,
        "dpar": dpar,
        "resp": resp,
        "loop": loop,
    }
    return FormulaConstruct._formula_parse(FormulaPart("nlf", formulas, formula_args))


def acformula(
    autocor: str,
    resp: str | None = None,
) -> FormulaConstruct:
    """
    Add an autocorrelation structure.

    Wraps R ``brms::acformula()``.

    Parameters
    ----------
    autocor : str
        One-sided autocorrelation formula (e.g. ``"~ arma(p = 1, q = 1)"``).
    resp : str or None, default=None
        Response name for multivariate models.

    Returns
    -------
    FormulaConstruct
        A composable formula specification.

    See Also
    --------
    brms::acformula : [R documentation](https://paulbuerkner.com/brms/reference/acformula.html)

    Examples
    --------
    ```python
    from brmspy.brms import bf, acformula

    f = bf("y ~ x") + acformula("~ arma(p = 1, q = 1)")
    ```
    """
    formula_args = {"resp": resp}
    return FormulaConstruct._formula_parse(
        FormulaPart("acformula", [autocor], formula_args)
    )


def set_rescor(rescor: bool = True) -> FormulaConstruct:
    """
    Control residual correlations in multivariate models.

    Wraps R ``brms::set_rescor()``.

    Parameters
    ----------
    rescor : bool, default=True
        Whether to model residual correlations.

    Returns
    -------
    FormulaConstruct
        A composable formula specification.

    See Also
    --------
    brms::set_rescor : [R documentation](https://paulbuerkner.com/brms/reference/set_rescor.html)

    Examples
    --------
    ```python
    from brmspy.brms import bf, set_rescor

    f = bf("y1 ~ x") + bf("y2 ~ z") + set_rescor(True)
    ```
    """
    formula_args = {
        "rescor": rescor,
    }
    return FormulaConstruct._formula_parse(FormulaPart("set_rescor", [], formula_args))


def set_mecor(mecor: bool = True) -> FormulaConstruct:
    """
    Control correlations between latent ``me()`` terms.

    Wraps R ``brms::set_mecor()``.

    Parameters
    ----------
    mecor : bool, default=True
        Whether to model correlations between latent variables introduced by ``me()``.

    Returns
    -------
    FormulaConstruct
        A composable formula specification.

    See Also
    --------
    brms::set_mecor : [R documentation](https://paulbuerkner.com/brms/reference/set_mecor.html)

    Examples
    --------
    ```python
    from brmspy.brms import bf, set_mecor

    f = bf("y ~ me(x, sdx)") + set_mecor(True)
    ```
    """
    formula_args = {
        "mecor": mecor,
    }
    return FormulaConstruct._formula_parse(FormulaPart("set_mecor", [], formula_args))


def set_nl(
    dpar: str | None = None,
    resp: str | None = None,
) -> FormulaConstruct:
    """
    Mark a model (or part of it) as non-linear.

    Wraps R ``brms::set_nl()``.

    Parameters
    ----------
    dpar : str or None, default=None
        Distributional parameter name (if only part of the model is non-linear).
    resp : str or None, default=None
        Response name for multivariate models.

    Returns
    -------
    FormulaConstruct
        A composable formula specification.

    See Also
    --------
    brms::set_nl : [R documentation](https://paulbuerkner.com/brms/reference/set_nl.html)

    Examples
    --------
    ```python
    from brmspy.brms import bf, lf, set_nl

    f = bf("y ~ a * inv_logit(x * b)") + lf("a + b ~ z") + set_nl()
    ```
    """
    formula_args = {
        "dpar": dpar,
        "resp": resp,
    }
    return FormulaConstruct._formula_parse(FormulaPart("set_nl", [], formula_args))


from typing import Callable, cast


from brmspy.types.brms_results import ProxyListSexpVector
from brmspy.types.formula_dsl import FormulaConstruct, FormulaPart
from rpy2.rinterface_lib.sexp import NULL, Sexp


def _execute_formula(formula: FormulaConstruct | Sexp | str) -> Sexp:
    import rpy2.robjects as ro

    if isinstance(formula, Sexp):
        return formula
    if isinstance(formula, str):
        formula = FormulaConstruct._formula_parse(formula)

    # Must run for formula functions, e.g me() to register
    ro.r("library(brms)")

    fun_add = cast(Callable[[Sexp, Sexp], Sexp], ro.r("function (a, b) a + b"))

    result: Sexp | None = None
    for summand in formula:
        subresult: Sexp = py_to_r(summand[0])
        for part in summand[1:]:
            subresult = fun_add(subresult, py_to_r(part))

        if result is None:
            result = subresult
        else:
            result = fun_add(result, subresult)

    assert result is not None
    return result
