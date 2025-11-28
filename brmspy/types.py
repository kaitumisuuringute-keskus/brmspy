"""Result types for brmspy functions."""

from dataclasses import dataclass
from typing import Any, Dict, Mapping, Optional
import rpy2.robjects as robjects
import arviz as az
import xarray as xr

@dataclass(frozen=True)
class PriorSpec:
    """
    Python representation of a single brms prior.

    Roughly corresponds to brms::prior_string() arguments.
    """
    prior: str
    class_: Optional[str] = None   # 'class' in brms: "b", "sd", "Intercept", ...
    coef: Optional[str] = None
    group: Optional[str] = None
    dpar: Optional[str] = None
    resp: Optional[str] = None
    nlpar: Optional[str] = None
    lb: Optional[float] = None
    ub: Optional[float] = None

    def to_brms_kwargs(self) -> Dict[str, Any]:
        """Map to kwargs for brms::prior_string()."""
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
    posterior: xr.Dataset
    posterior_predictive: xr.Dataset
    log_likelihood: xr.Dataset
    observed_data: xr.Dataset
    coords: xr.Dataset
    dims: xr.Dataset

class IDEpred(az.InferenceData):
    posterior: xr.Dataset

class IDPredict(az.InferenceData):
    posterior_predictive: xr.Dataset

class IDLinpred(az.InferenceData):
    predictions: xr.Dataset

class IDLogLik(az.InferenceData):
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
    idata: IDLogLik
    r: robjects.ListVector

@dataclass
class PosteriorLinpredResult:
    idata: IDLinpred
    r: robjects.ListVector

@dataclass
class FormulaResult:
    r: robjects.ListVector
    dict: Dict