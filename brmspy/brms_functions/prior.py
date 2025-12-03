


from typing import Any, Optional

from brmspy.types import PriorSpec


def prior(
    prior: str,
    class_: Optional[str] = None,
    coef: Optional[str] = None,
    group: Optional[str] = None,
    dpar: Optional[str] = None,
    resp: Optional[str] = None,
    nlpar: Optional[str] = None,
    lb: Optional[float] = None,
    ub: Optional[float] = None,
    **kwargs: Any
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
        A typed prior specification object used by ``brmspy.fit()`` and
        ``brmspy.make_stancode()``.

    Notes
    -----
    This function does **not** validate the prior expression string itself —
    validation occurs inside brms. Its purpose is to construct a structured,
    editor-friendly representation that seamlessly maps to rpy2 calls.

    Examples
    --------
    Prior on the intercept ::

        p = prior("student_t(3, 0, 1.95)", class_="Intercept")

    Prior on a coefficient ::

        p = prior("normal(0, 1)", class_="b", coef="age")

    Group-level (hierarchical) SD prior ::

        p = prior("exponential(2)", class_="sd", group="region")

    Truncated prior ::

        p = prior("normal(0, 1)", class_="b", coef="income", lb=0)

    Multiple priors passed to ``fit`` ::

        from brmspy import prior
        priors = [
            prior("student_t(3, 0, 2)", class_="b", coef="zAge"),
            prior("exponential(2)", class_="sd", group="patient"),
        ]
        model = brms.fit("y ~ zAge + (1|patient)", data=df, priors=priors)
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
        **kwargs
    )