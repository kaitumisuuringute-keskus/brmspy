from typing import Union, cast
import rpy2.robjects as ro

from ..runtime._state import get_brms
from ..helpers.conversion import (
    kwargs_r, py_to_r, r_to_py
)
from ..types import (
    FormulaResult
)


def _formula_parse(r: ro.ListVector) -> FormulaResult:
    return FormulaResult(r=r, dict=cast(dict, r_to_py(r)))

def formula(
    formula: str,
    **formula_args
) -> FormulaResult:
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
        f = brms.formula("y ~ x1 + x2 + (1|group)")
    ```

    With QR decomposition for numerical stability:
    
    ```python
    f = brms.formula(
        "reaction ~ days + (days|subject)",
        decomp="QR"
    )
    ```

    Multivariate formula:
    ```
        f = brms.formula(
            "mvbind(y1, y2) ~ x1 + x2",
            center=True
        )
    ```
    """
    brms = get_brms()
    formula_args = kwargs_r(formula_args)
    formula_obj = brms.bf(formula, **formula_args)
    return _formula_parse(formula_obj)




def set_rescor(formula: Union[FormulaResult, ro.ListVector], rescor: bool=True):
    brms = get_brms()
    formula_args = kwargs_r({
        "rescor": rescor
    })
    formula_obj = brms.set_rescor(py_to_r(formula), **formula_args)
    return _formula_parse(formula_obj)

def set_mecor(formula: Union[FormulaResult, ro.ListVector], mecor: bool=True):
    brms = get_brms()
    formula_args = kwargs_r({
        "mecor": mecor
    })
    formula_obj = brms.set_mecor(py_to_r(formula), **formula_args)
    return _formula_parse(formula_obj)

def set_nl(formula: Union[FormulaResult, ro.ListVector], dpar=None, resp=None):
    brms = get_brms()
    formula_args = kwargs_r({
        "dpar": dpar,
        "resp": resp
    })
    formula_obj = brms.set_nl(py_to_r(formula), **formula_args)
    return _formula_parse(formula_obj)
