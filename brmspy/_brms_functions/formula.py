from collections.abc import Callable
from typing import cast, get_args
from rpy2.rinterface_lib.sexp import Sexp
from brmspy.helpers.log import log

from brmspy.types.session import SexpWrapper
from ..helpers._rpy2._conversion import kwargs_r, py_to_r
from ..types.formula_dsl import (
    _FORMULA_FUNCTION_WHITELIST,
    FormulaConstruct,
    FormulaPart,
)

from ..types.brms_results import ProxyListSexpVector


def bf(*formulas: str, **formula_args) -> FormulaConstruct:
    """
    Set up a model formula for brms package.

    Allows defining (potentially non-linear) additive multilevel models
    for all parameters of the assumed response distribution.

    Parameters
    ----------
    formula : str
        brms formula specification, e.g., "y ~ x + (1|group)"
    **formula_args : dict
        Additional brms::brmsformula() arguments:

        - decomp : str
            Decomposition method (e.g., "QR" for QR decomposition)
        - center : bool
            Whether to center predictors (default True)
        - sparse : bool
            Use sparse matrix representation
        - nl : bool
            Whether formula is non-linear
        - loop : bool
            Use loop-based Stan code

    Returns
    -------
    FormulaResult
        Object with .r (R brmsformula object) and .dict (Python dict) attributes

    See Also
    --------
    brms::brmsformula : R documentation
        https://paulbuerkner.com/brms/reference/brmsformula.html
    fit : Fit model using formula

    Examples
    --------
    Basic formula:

    ```python
        from brmspy import brms
        f = brms.bf("y ~ x1 + x2 + (1|group)")
    ```

    With QR decomposition for numerical stability:

    ```python
    f = brms.bf(
        "reaction ~ days + (days|subject)",
        decomp="QR"
    )
    ```

    Multivariate formula:
    ```
        f = brms.bf(
            "mvbind(y1, y2) ~ x1 + x2",
            center=True
        )
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
    Specify linear formulas for distributional / non-linear parameters.

    Parameters
    ----------
    *formulas : str or FormulaResult
        One or more formulas (e.g. "sigma ~ x", "nu ~ z").
    flist : list, optional
        Additional formulas passed as a list.
    dpar : str, optional
        Name of the distributional parameter (e.g. "sigma").
    resp : str, optional
        Response name in multivariate models.
    center, cmc, sparse, decomp : optional
        Passed through to brms::lf().

    Returns
    -------
    FormulaResult
        Object that can be added to a bf() formula via +.

    Examples
    --------
    >>> f = bf("y ~ 1") + lf("sigma ~ x", dpar="sigma")
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
    Specify non-linear formulas for distributional parameters.

    Parameters
    ----------
    formula : str
        Non-linear formula, e.g. "sigma ~ a * exp(b * x)".
    *extra : str or FormulaResult
        Additional named parameters or formulas (rarely needed here;
        typically you use lf() for those).
    flist : list, optional
        Additional formulas passed as a list.
    dpar : str, optional
        Name of the distributional parameter.
    resp : str, optional
        Response name in multivariate models.
    loop : bool, optional
        Whether to compute inside a loop (brms::nlf(loop=...)).

    Returns
    -------
    FormulaResult
        Object that can be added to a bf() formula via +.

    Examples
    --------
    >>> f = bf("y ~ 1") + nlf("sigma ~ a * exp(b * x)")
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
    Specify autocorrelation terms to add to a model.

    Parameters
    ----------
    autocor : str
        One-sided formula with autocorrelation terms,
        e.g. "~ arma(p = 1, q = 1)".
    resp : str, optional
        Response name in multivariate models.

    Returns
    -------
    FormulaResult
        Object that can be added to a bf() formula via +.

    Examples
    --------
    >>> f = bf("y ~ x") + acformula("~ arma(p = 1, q = 1)")
    """
    formula_args = {"resp": resp}
    return FormulaConstruct._formula_parse(
        FormulaPart("acformula", [autocor], formula_args)
    )


def set_rescor(rescor: bool = True) -> FormulaConstruct:
    """
    Control residual correlations in multivariate models.

    Parameters
    ----------
    rescor : bool, default True
        Whether to model residual correlations.

    Returns
    -------
    FormulaResult
        Object that can be added to a multivariate formula via +.

    Examples
    --------
    >>> f = bf("y1 ~ x") + bf("y2 ~ z") + set_rescor(True)
    """
    formula_args = {
        "rescor": rescor,
    }
    return FormulaConstruct._formula_parse(FormulaPart("set_rescor", [], formula_args))


def set_mecor(mecor: bool = True) -> FormulaConstruct:
    """
    Control correlations between latent me-terms.

    Parameters
    ----------
    mecor : bool, default True
        Whether to model correlations between me() latent variables.

    Returns
    -------
    FormulaResult
        Object that can be added to a formula via +.

    Examples
    --------
    >>> f = bf("y ~ me(x, sdx)") + set_mecor(True)
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
    Mark a formula as non-linear (or parts of it).

    Parameters
    ----------
    dpar : str, optional
        Distributional parameter name (if only part of the model is non-linear).
    resp : str, optional
        Response name in multivariate models.

    Returns
    -------
    FormulaResult
        Object that can be added to a formula via +.

    Examples
    --------
    >>> f = bf("y ~ a * inv_logit(x * b)") + lf("a + b ~ z") + set_nl()
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
