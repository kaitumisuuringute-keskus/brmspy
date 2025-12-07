
import typing
import pandas as pd
from .formula import bf
from ..helpers.priors import _build_priors
from ..runtime._state import get_brms
from ..helpers.conversion import (
    py_to_r
)
from ..types import (
    FormulaResult, PriorSpec
)


_formula_fn = bf

def make_stancode(
    formula: typing.Union[FormulaResult, str],
    data: pd.DataFrame,
    priors: typing.Optional[typing.Sequence[PriorSpec]] = None,
    family: str = "poisson",
    sample_prior: str = "no",
    formula_args: typing.Optional[dict] = None
) -> str:
    """
    Generate Stan code using brms::make_stancode().
    
    Useful for inspecting the generated Stan model before fitting,
    understanding the model structure, or using the code with other
    Stan interfaces.
    
    Parameters
    ----------
    formula : str or FormulaResult
        brms formula specification
    data : pd.DataFrame
        Model data
    priors : list of PriorSpec, optional
        Prior specifications from prior() function
    family : str, default="poisson"
        Distribution family (gaussian, poisson, binomial, etc.)
    sample_prior : str, default="no"
        Whether to sample from prior:
        - "no": No prior samples
        - "yes": Include prior samples alongside posterior
        - "only": Sample from prior only (no data)
    formula_args : dict, optional
        Additional arguments passed to formula()
    
    Returns
    -------
    str
        Complete Stan program code as string
    
    See Also
    --------
    brms::make_stancode : R documentation
        https://paulbuerkner.com/brms/reference/make_stancode.html
    fit : Fit model instead of just generating code
    make_standata : Generate Stan data block
    
    Examples
    --------
    Generate Stan code for simple model:
    
    ```python
    from brmspy import brms
    epilepsy = brms.get_brms_data("epilepsy")
    
    stan_code = brms.make_stancode(
        formula="count ~ zAge + zBase * Trt + (1|patient)",
        data=epilepsy,
        family="poisson"
    )
    
    print(stan_code[:500])  # Print first 500 characters
    ```

    With custom priors:
    
    ```python
        from brmspy import prior
        
        stan_code = brms.make_stancode(
            formula="count ~ zAge",
            data=epilepsy,
            priors=[prior("normal(0, 1)", class_="b")],
            family="poisson"
        )
    ```

    For prior predictive checks (sample_prior="only"):

    ```
    stan_code = brms.make_stancode(
        formula="count ~ zAge",
        data=epilepsy,
        family="poisson",
        sample_prior="only"
    )
    ```
    """
    brms = get_brms()

    data_r = py_to_r(data)
    priors_r = _build_priors(priors)
    if isinstance(formula, FormulaResult):
        formula_obj = formula.r
    else:
        if formula_args is None:
            formula_args = {}
        formula_obj = _formula_fn(formula, **formula_args).r


    if len(priors_r) > 0:
        return brms.make_stancode(
            formula=formula_obj, data=data_r, prior=priors_r, family=family, sample_prior=sample_prior
        )[0]
    else:
        return brms.make_stancode(
            formula=formula_obj, data=data_r, family=family, sample_prior=sample_prior
        )[0]

