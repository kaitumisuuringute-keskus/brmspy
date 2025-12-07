
import typing

from brmspy.runtime._state import get_brms
from brmspy.types import PriorSpec


def _build_priors(priors: typing.Optional[typing.Sequence[PriorSpec]] = None) -> list:
    """
    Build R brms prior object from Python PriorSpec specifications.
    
    Converts a sequence of PriorSpec objects to a single combined R brms prior
    object by calling brms::prior_string() for each spec and combining with `+`.
    Used internally by fit() to translate Python prior specifications to R.
    
    Parameters
    ----------
    priors : sequence of PriorSpec, optional
        List of prior specifications. Each PriorSpec contains:
        - prior: Prior distribution string (e.g., "normal(0, 1)")
        - class_: Parameter class (e.g., "b", "Intercept", "sigma")
        - coef: Specific coefficient name (optional)
        - group: Group-level effects (optional)
        
        If None or empty, returns empty list (brms uses default priors)
    
    Returns
    -------
    R brmsprior object or list
        Combined R brms prior object if priors provided, empty list otherwise
    
    Raises
    ------
    AssertionError
        If combined result is not a valid brmsprior object
    
    Notes
    -----
    **Prior Combination:**
    
    Multiple priors are combined using R's `+` operator:
    ```R
    prior1 + prior2 + prior3
    ```
    
    This creates a single brmsprior object containing all specifications.
    
    **brms Prior Classes:**
    
    Common parameter classes:
    - **b**: Population-level effects (regression coefficients)
    - **Intercept**: Model intercept
    - **sigma**: Residual standard deviation (for gaussian family)
    - **sd**: Standard deviation of group-level effects
    - **cor**: Correlation of group-level effects
    
    **Prior String Format:**
    
    brms uses Stan-style prior specifications:
    - Normal: "normal(mean, sd)"
    - Student-t: "student_t(df, location, scale)"
    - Cauchy: "cauchy(location, scale)"
    - Exponential: "exponential(rate)"
    - Uniform: "uniform(lower, upper)"
    
    Examples
    --------

    ```python
    from brmspy.types import PriorSpec
    from brmspy.helpers.priors import _build_priors
    
    # Single prior for regression coefficients
    priors = [
        PriorSpec(
            prior="normal(0, 1)",
            class_="b"
        )
    ]
    brms_prior = _build_priors(priors)
    ```

    See Also
    --------
    brmspy.types.PriorSpec : Prior specification class
    brmspy.brms.fit : Uses this to convert priors for model fitting
    brms::prior : R brms prior specification
    brms::set_prior : R function for setting priors
    
    References
    ----------
    .. [1] brms prior documentation: https://paul-buerkner.github.io/brms/reference/set_prior.html
    .. [2] Stan prior choice recommendations: https://github.com/stan-dev/stan/wiki/Prior-Choice-Recommendations
    """
    brms = get_brms()
    if not priors:
        return []

    prior_objs = []
    for p in priors:
        kwargs = p.to_brms_kwargs()
        # first argument is the prior string
        prior_str = kwargs.pop("prior")
        prior_obj = brms.prior_string(prior_str, **kwargs)
        prior_objs.append(prior_obj)

    brms_prior = prior_objs[0]
    for p in prior_objs[1:]:
        brms_prior = brms_prior + p

    assert brms.is_brmsprior(brms_prior)
    return brms_prior