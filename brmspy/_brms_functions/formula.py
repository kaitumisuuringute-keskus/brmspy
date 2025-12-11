from collections.abc import Callable
from typing import cast

from ..helpers._rpy2._conversion import kwargs_r, py_to_r
from ..types.brms_results import FormulaResult


def bf(formula: str, **formula_args) -> FormulaResult:
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
    import rpy2.robjects as ro

    fun = cast(Callable, ro.r("brms::bf"))
    formula_args = kwargs_r(formula_args)
    formula_obj = fun(formula, **formula_args)
    return FormulaResult._formula_parse(formula_obj)


def lf(
    *formulas: str | FormulaResult | object,
    flist=None,
    dpar: str | None = None,
    resp: str | None = None,
    center: bool | None = None,
    cmc: bool | None = None,
    sparse: bool | None = None,
    decomp: str | None = None,
) -> FormulaResult:
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
    import rpy2.robjects as ro

    fun = cast(Callable, ro.r("brms::lf"))

    r_formulas = [py_to_r(f) for f in formulas]

    r_flist = None
    if flist is not None:
        r_flist = [py_to_r(f) for f in flist]

    formula_args = kwargs_r(
        {
            "flist": r_flist,
            "dpar": dpar,
            "resp": resp,
            "center": center,
            "cmc": cmc,
            "sparse": sparse,
            "decomp": decomp,
        }
    )

    formula_obj = fun(*r_formulas, **formula_args)
    return FormulaResult._formula_parse(formula_obj)


def nlf(
    formula: str | object,
    *extra: str | FormulaResult | object,
    flist=None,
    dpar: str | None = None,
    resp: str | None = None,
    loop: bool | None = None,
) -> FormulaResult:
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
    import rpy2.robjects as ro

    fun = cast(Callable, ro.r("brms::nlf"))

    r_formula = py_to_r(formula)
    r_extra = [py_to_r(x) for x in extra]

    r_flist = None
    if flist is not None:
        r_flist = [py_to_r(f) for f in flist]

    formula_args = kwargs_r(
        {
            "flist": r_flist,
            "dpar": dpar,
            "resp": resp,
            "loop": loop,
        }
    )

    formula_obj = fun(r_formula, *r_extra, **formula_args)
    return FormulaResult._formula_parse(formula_obj)


def acformula(
    autocor: str | object,
    resp: str | None = None,
) -> FormulaResult:
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
    import rpy2.robjects as ro

    fun = cast(Callable, ro.r("brms::acformula"))
    r_autocor = py_to_r(autocor)

    formula_args = kwargs_r(
        {
            "resp": resp,
        }
    )

    formula_obj = fun(r_autocor, **formula_args)
    return FormulaResult._formula_parse(formula_obj)


def set_rescor(rescor: bool = True) -> FormulaResult:
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
    import rpy2.robjects as ro

    fun = cast(Callable, ro.r("brms::set_rescor"))
    formula_args = kwargs_r(
        {
            "rescor": rescor,
        }
    )
    formula_obj = fun(**formula_args)
    return FormulaResult._formula_parse(formula_obj)


def set_mecor(mecor: bool = True) -> FormulaResult:
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
    import rpy2.robjects as ro

    fun = cast(Callable, ro.r("brms::set_mecor"))
    formula_args = kwargs_r(
        {
            "mecor": mecor,
        }
    )
    formula_obj = fun(**formula_args)
    return FormulaResult._formula_parse(formula_obj)


def set_nl(
    dpar: str | None = None,
    resp: str | None = None,
) -> FormulaResult:
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
    import rpy2.robjects as ro

    fun = cast(Callable, ro.r("brms::set_nl"))
    formula_args = kwargs_r(
        {
            "dpar": dpar,
            "resp": resp,
        }
    )
    formula_obj = fun(**formula_args)
    return FormulaResult._formula_parse(formula_obj)


def _formula_add(a: FormulaResult, b: FormulaResult) -> FormulaResult:
    import rpy2.robjects as ro

    r_fun = cast(Callable, ro.r("function (a, b) a + b"))

    r = r_fun(a.r, b.r)

    return FormulaResult(parts=[a, b], r=r)
