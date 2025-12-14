"""
Prior specification helpers.

This module provides helpers for constructing brms-compatible prior
specifications and for querying the default priors implied by a model.

Notes
-----
Executed inside the worker process that hosts the embedded R session.
"""

from collections.abc import Callable
from typing import Any, cast

import pandas as pd
from rpy2.rinterface import ListSexpVector

from brmspy._brms_functions.formula import _execute_formula
from brmspy.helpers._rpy2._conversion import kwargs_r, py_to_r, r_to_py
from brmspy.types.brms_results import PriorSpec, RListVectorExtension
from brmspy.types.formula_dsl import FormulaConstruct


def prior(
    prior: str,
    class_: str | None = None,
    coef: str | None = None,
    group: str | None = None,
    dpar: str | None = None,
    resp: str | None = None,
    nlpar: str | None = None,
    lb: float | None = None,
    ub: float | None = None,
    **kwargs: Any,
) -> PriorSpec:
    """
    Create a brms-style prior specification.

    This function mirrors the behavior of ``brms::prior_string()`` and allows
    specifying priors for regression parameters, group-level effects, nonlinear
    parameters, distributional parameters, and more — using a typed Python
    interface. All arguments correspond directly to the parameters of
    ``prior_string()`` in brms.

    Parameters
    ----------
    prior : str
        The prior definition as a string, exactly as brms expects it.
        Examples include ::

            "normal(0, 1)"
            "student_t(3, 0, 1.5)"
            "exponential(2)"
            "lkj(2)"

    class_ : str, optional
        Parameter class (e.g. ``"b"``, ``"sd"``, ``"Intercept"``).
        This corresponds to ``class`` in brms. ``class`` cannot be used as a
        parameter in Python (reserved keyword), so ``class_`` is used instead.

    coef : str, optional
        Coefficient name for class-level effects.

    group : str, optional
        Grouping variable for hierarchical/multilevel effects.

    dpar : str, optional
        Distributional parameter (e.g. ``"sigma"`` or ``"phi"``).

    resp : str, optional
        Response variable name for multivariate models.

    nlpar : str, optional
        Nonlinear parameter name if using nonlinear formulas.

    lb : float, optional
        Lower bound for truncated priors.

    ub : float, optional
        Upper bound for truncated priors.

    **kwargs
        Any additional keyword arguments supported by ``brms::prior_string()``.
        These are forwarded unchanged.

    Returns
    -------
    PriorSpec
        A typed prior specification object used by `brmspy.brms.brm()` and
        `brmspy.brms.make_stancode()`.

    See Also
    --------
    brms::prior_string : [R documentation](https://paulbuerkner.com/brms/reference/prior_string.html)

    Notes
    -----
    This function does **not** validate the prior expression string itself —
    validation occurs inside brms.

    Examples
    --------
    ```python
    from brmspy.brms import prior

    p_intercept = prior("student_t(3, 0, 1.95)", class_="Intercept")
    p_slope = prior("normal(0, 1)", class_="b", coef="age")
    p_sd = prior("exponential(2)", class_="sd", group="region")
    p_trunc = prior("normal(0, 1)", class_="b", coef="income", lb=0)
    ```
    """
    if "class" in kwargs:
        kwargs["class_"] = kwargs["class"]

    return PriorSpec(
        prior=prior,
        class_=class_,
        coef=coef,
        group=group,
        dpar=dpar,
        resp=resp,
        nlpar=nlpar,
        lb=lb,
        ub=ub,
        **kwargs,
    )


def get_prior(
    formula: str | FormulaConstruct, data=None, family="gaussian", **kwargs
) -> pd.DataFrame:
    """
    Get default priors for a model specification.

    Wrapper around R ``brms::get_prior()``.

    Returns a DataFrame with default priors for each parameter class in the specified
    brms model. Useful for reviewing and customizing priors before fitting.

    Parameters
    ----------
    formula : str or FormulaConstruct
        Model formula (e.g. ``"y ~ x + (1|group)"``) or a composed formula.
    data : pd.DataFrame or dict, optional
        Dataset containing model variables. Required for data-dependent priors
    family : str or ListSexpVector, default="gaussian"
        Distribution family (e.g., "gaussian", "poisson", "binomial")
    **kwargs
        Additional arguments passed to brms::get_prior()
        (e.g., autocor, data2, knots, drop_unused_levels)

    Returns
    -------
    pd.DataFrame
        DataFrame with columns: prior, class, coef, group, resp, dpar, nlpar, lb, ub, source.
        Each row represents a parameter or parameter class that can have a custom prior.

    See Also
    --------
    default_prior : Generic function for getting default priors
    prior : Create custom prior specifications
    brms::get_prior : [R documentation](https://paulbuerkner.com/brms/reference/get_prior.html)

    Examples
    --------
    ```python
    from brmspy import brms
    from brmspy.brms import prior

    priors_df = brms.get_prior("y ~ x", data=df)

    custom_priors = [
        prior("normal(0, 0.5)", class_="b"),
        prior("exponential(2)", class_="sigma"),
    ]

    fit = brms.brm("y ~ x", data=df, priors=custom_priors, chains=4)
    ```
    """
    import rpy2.robjects as ro

    formula_obj = _execute_formula(formula)

    r_get_prior = cast(Callable, ro.r("brms::get_prior"))
    collected_args = kwargs_r(
        {"formula": formula_obj, "data": data, "family": family, **kwargs}
    )

    df_r = r_get_prior(**collected_args)
    df = pd.DataFrame(cast(Any, r_to_py(df_r)))

    return df


def default_prior(
    object: RListVectorExtension | ListSexpVector | FormulaConstruct | str,
    data=None,
    family="gaussian",
    **kwargs,
) -> pd.DataFrame:
    """
    Get default priors for brms model parameters (generic function).

    Wrapper around R ``brms::default_prior()``.

    Generic function to retrieve default prior specifications for all parameters
    in a brms model. Accepts formula objects, brmsformula objects, or other model
    specification objects. This is the generic version of get_prior().

    Parameters
    ----------
    object : str, FormulaResult, or ListSexpVector
        Model specification: formula string, brmsformula object, mvbrmsformula,
        or any object that can be coerced to these classes
    data : pd.DataFrame or dict, optional
        Dataset containing model variables. Required for data-dependent priors
    family : str or ListSexpVector, default="gaussian"
        Distribution family (e.g., "gaussian", "poisson", "binomial").
        Can be a list of families for multivariate models
    **kwargs
        Additional arguments passed to brms::get_prior()
        (e.g., autocor, data2, knots, drop_unused_levels, sparse)

    Returns
    -------
    pd.DataFrame
        DataFrame with columns: prior, class, coef, group, resp, dpar, nlpar, lb, ub, source.
        Each row specifies a parameter class with its default prior. The 'prior' column
        is empty except for internal defaults.

    See Also
    --------
    get_prior : Convenience function with formula parameter
    prior : Create custom prior specifications
    brms::default_prior : [R documentation](https://paulbuerkner.com/brms/reference/default_prior.html)

    Examples
    --------
    Get default priors for a Poisson model:

    ```python
    from brmspy import brms

    priors = brms.default_prior(
        object="count ~ zAge + zBase * Trt + (1|patient)",
        data=epilepsy,
        family="poisson"
    )
    print(priors)
    ```

    Use with formula object:

    ```python
    from brmspy import brms

    f = brms.formula("y ~ x + (1|group)")
    priors = brms.default_prior(f, data=df, family="gaussian")
    ```
    """
    import rpy2.robjects as ro

    r_get_prior = cast(Callable, ro.r("brms::get_prior"))
    collected_args = kwargs_r({"data": data, "family": family, **kwargs})

    obj_resolved = object
    if isinstance(object, FormulaConstruct):
        obj_resolved = _execute_formula(object)

    df_r = r_get_prior(py_to_r(obj_resolved), **collected_args)
    df = pd.DataFrame(cast(Any, r_to_py(df_r)))

    return df
