"""Result types for brmspy functions."""

from dataclasses import dataclass
from typing import Any, Dict, Mapping, Optional
import rpy2.robjects as robjects
import arviz as az
import xarray as xr

@dataclass(frozen=True)
class PriorSpec:
    """
    Python representation of a brms prior specification.
    
    This dataclass provides a typed interface to brms::prior_string() arguments,
    allowing Python developers to specify priors with IDE autocomplete and
    type checking. Use the `prior()` factory function to create instances.
    
    Attributes
    ----------
    prior : str
        Prior distribution as string (e.g., "normal(0, 1)", "exponential(2)")
    class_ : str, optional
        Parameter class: "b" (fixed effects), "sd" (group SD),
        "Intercept", "sigma", "cor", etc.
    coef : str, optional
        Specific coefficient name for class-level priors
    group : str, optional
        Grouping variable for hierarchical effects
    dpar : str, optional
        Distributional parameter (e.g., "sigma", "phi", "zi")
    resp : str, optional
        Response variable for multivariate models
    nlpar : str, optional
        Non-linear parameter name
    lb : float, optional
        Lower bound for truncated priors
    ub : float, optional
        Upper bound for truncated priors
    
    See Also
    --------
    prior : Factory function to create PriorSpec instances
    brms::prior_string : R documentation
        https://paulbuerkner.com/brms/reference/prior_string.html
    
    Examples
    --------
    Create prior specifications (prefer using `prior()` function):
    
    ```python
    from brmspy.types import PriorSpec
    
    # Fixed effect prior
    p1 = PriorSpec(prior="normal(0, 1)", class_="b")
    
    # Group-level SD prior
    p2 = PriorSpec(
        prior="exponential(2)",
        class_="sd",
        group="patient"
    )
    
    # Coefficient-specific prior with bounds
    p3 = PriorSpec(
        prior="normal(0, 1)",
        class_="b",
        coef="age",
        lb=0  # Truncated at zero
    )
    ```
    """
    prior: str
    class_: Optional[str] = None
    coef: Optional[str] = None
    group: Optional[str] = None
    dpar: Optional[str] = None
    resp: Optional[str] = None
    nlpar: Optional[str] = None
    lb: Optional[float] = None
    ub: Optional[float] = None

    def to_brms_kwargs(self) -> Dict[str, Any]:
        """
        Convert PriorSpec to keyword arguments for brms::prior_string().
        
        Maps Python dataclass fields to R function arguments, handling
        the `class_` -> `class` parameter name conversion.
        
        Returns
        -------
        dict
            Keyword arguments ready for brms::prior_string()
        
        Examples
        --------
        ```python
        from brmspy import prior
        p = prior("normal(0, 1)", class_="b", coef="age")
        kwargs = p.to_brms_kwargs()
        print(kwargs)
        # {'prior': 'normal(0, 1)', 'class': 'b', 'coef': 'age'}
        ```
        """
        out: dict[str, Any] = {"prior": self.prior}
        if self.class_ is not None:
            out["class"] = self.class_
        if self.coef is not None:
            out["coef"] = self.coef
        if self.group is not None:
            out["group"] = self.group
        if self.dpar is not None:
            out["dpar"] = self.dpar
        if self.resp is not None:
            out["resp"] = self.resp
        if self.nlpar is not None:
            out["nlpar"] = self.nlpar
        if self.lb is not None:
            out["lb"] = self.lb
        if self.ub is not None:
            out["ub"] = self.ub
        return out

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


# -----------------------------------------------------
# az.InferenceData extensions for proper typing in IDEs
# -----------------------------------------------------

class IDFit(az.InferenceData):
    """
    Typed InferenceData for fitted brms models.
    
    Extends arviz.InferenceData with type hints for IDE autocomplete.
    Guarantees the presence of specific data groups returned by `fit()`.
    
    Attributes
    ----------
    posterior : xr.Dataset
        Posterior samples of model parameters
    posterior_predictive : xr.Dataset
        Posterior predictive samples (with observation noise)
    log_likelihood : xr.Dataset
        Log-likelihood values for each observation
    observed_data : xr.Dataset
        Original observed response data
    coords : dict
        Coordinate mappings for dimensions
    dims : dict
        Dimension specifications for variables
    
    See Also
    --------
    brmspy.brms.fit : Creates IDFit objects
    arviz.InferenceData : Base class documentation
    
    Examples
    --------

    ```python
    from brmspy import brms
    model = brms.fit("y ~ x", data=df, chains=4)
    
    # Type checking and autocomplete
    assert isinstance(model.idata, IDFit)
    print(model.idata.posterior)  # IDE autocomplete works!
    ```

    """
    posterior: xr.Dataset
    posterior_predictive: xr.Dataset
    log_likelihood: xr.Dataset
    observed_data: xr.Dataset
    coords: xr.Dataset
    dims: xr.Dataset

class IDEpred(az.InferenceData):
    """
    Typed InferenceData for posterior_epred results.
    
    Contains expected values E[Y|X] without observation noise.
    
    Attributes
    ----------
    posterior : xr.Dataset
        Expected value samples (no observation noise)
    
    See Also
    --------
    brmspy.brms.posterior_epred : Creates IDEpred objects
    """
    posterior: xr.Dataset

class IDPredict(az.InferenceData):
    """
    Typed InferenceData for posterior_predict results.
    
    Contains posterior predictive samples with observation noise.
    
    Attributes
    ----------
    posterior_predictive : xr.Dataset
        Posterior predictive samples (includes observation noise)
    
    See Also
    --------
    brmspy.brms.posterior_predict : Creates IDPredict objects
    """
    posterior_predictive: xr.Dataset

class IDLinpred(az.InferenceData):
    """
    Typed InferenceData for posterior_linpred results.
    
    Contains linear predictor values (before applying link function).
    
    Attributes
    ----------
    predictions : xr.Dataset
        Linear predictor samples
    
    See Also
    --------
    brmspy.brms.posterior_linpred : Creates IDLinpred objects
    """
    predictions: xr.Dataset

class IDLogLik(az.InferenceData):
    """
    Typed InferenceData for log_lik results.
    
    Contains log-likelihood values for model comparison.
    
    Attributes
    ----------
    log_likelihood : xr.Dataset
        Log-likelihood values for each observation
    
    See Also
    --------
    brmspy.brms.log_lik : Creates IDLogLik objects
    arviz.loo : LOO-CV using log-likelihood
    """
    log_likelihood: xr.Dataset




# ---------------------
# Function return types
# ---------------------

@dataclass
class GenericResult:
    """Generic result container with arviz and R objects.
    
    Attributes
    ----------
    idata : arviz.InferenceData
        arviz InferenceData object
    r : robjects.ListVector
        R object from brms
    """
    idata: az.InferenceData
    r: robjects.ListVector

@dataclass
class FitResult:
    """Result from fit() function.
    
    Attributes
    ----------
    idata : arviz.InferenceData
        arviz InferenceData with posterior, posterior_predictive,
        log_likelihood, and observed_data groups
    r : robjects.ListVector
        brmsfit R object from brms::brm()
    """
    idata: IDFit
    r: robjects.ListVector

@dataclass
class PosteriorEpredResult:
    """Result from posterior_epred() function.
    
    Attributes
    ----------
    idata : arviz.InferenceData
        arviz InferenceData with expected values in 'posterior' group
    r : robjects.ListVector
        R matrix from brms::posterior_epred()
    """
    idata: IDEpred
    r: robjects.ListVector

@dataclass
class PosteriorPredictResult:
    """Result from posterior_predict() function.
    
    Attributes
    ----------
    idata : arviz.InferenceData
        arviz InferenceData with predictions in 'posterior_predictive' group
    r : robjects.ListVector
        R matrix from brms::posterior_predict()
    """
    idata: IDPredict
    r: robjects.ListVector

@dataclass
class LogLikResult:
    """
    Result from log_lik() function.
    
    Attributes
    ----------
    idata : IDLogLik
        arviz InferenceData with log-likelihood values
    r : robjects.ListVector
        R matrix from brms::log_lik()
    
    See Also
    --------
    brmspy.brms.log_lik : Creates LogLikResult objects
    
    Examples
    --------

    ```python
    from brmspy import brms
    import arviz as az
    
    model = brms.fit("y ~ x", data=df, chains=4)
    loglik = brms.log_lik(model)
    
    # Use for model comparison
    loo = az.loo(model.idata)
    print(loo)
    ```

    """
    idata: IDLogLik
    r: robjects.ListVector

@dataclass
class PosteriorLinpredResult:
    """
    Result from posterior_linpred() function.
    
    Attributes
    ----------
    idata : IDLinpred
        arviz InferenceData with linear predictor values
    r : robjects.ListVector
        R matrix from brms::posterior_linpred()
    
    See Also
    --------
    brmspy.brms.posterior_linpred : Creates PosteriorLinpredResult objects
    
    Examples
    --------

    ```python
    from brmspy import brms
    
    model = brms.fit("count ~ age", data=df, family="poisson", chains=4)
    linpred = brms.posterior_linpred(model)
    
    # Linear predictor on log scale (for Poisson)
    print(linpred.idata.predictions)
    ```
    """
    idata: IDLinpred
    r: robjects.ListVector

@dataclass
class FormulaResult:
    """
    Result from formula() function.
    
    Attributes
    ----------
    r : robjects.ListVector
        R brmsformula object
    dict : Dict
        Python dictionary representation of formula
    
    See Also
    --------
    brmspy.brms.formula : Creates FormulaResult objects
    
    Examples
    --------

    ```python
    from brmspy import brms
    
    # Create formula with options
    f = brms.formula("y ~ x", decomp="QR")
    
    # Use in fit()
    model = brms.fit(f, data=df, chains=4)
    ```
    """
    r: robjects.ListVector
    dict: Dict