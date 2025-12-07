"""Result types for brmspy functions."""

from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Mapping, Optional, Union, cast
import rpy2.robjects as robjects
import arviz as az
import xarray as xr
import pandas as pd


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
class RListVectorExtension:
    """Generic result container with R objects.
    
    Attributes
    ----------
    r : robjects.ListVector
        R object from brms
    """
    r: robjects.ListVector

@dataclass
class GenericResult(RListVectorExtension):
    """Generic result container with arviz and R objects.
    
    Attributes
    ----------
    idata : arviz.InferenceData
        arviz InferenceData object
    r : robjects.ListVector
        R object from brms
    """
    idata: az.InferenceData

@dataclass
class FitResult(RListVectorExtension):
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

@dataclass
class PosteriorEpredResult(RListVectorExtension):
    """Result from posterior_epred() function.
    
    Attributes
    ----------
    idata : arviz.InferenceData
        arviz InferenceData with expected values in 'posterior' group
    r : robjects.ListVector
        R matrix from brms::posterior_epred()
    """
    idata: IDEpred

@dataclass
class PosteriorPredictResult(RListVectorExtension):
    """Result from posterior_predict() function.
    
    Attributes
    ----------
    idata : arviz.InferenceData
        arviz InferenceData with predictions in 'posterior_predictive' group
    r : robjects.ListVector
        R matrix from brms::posterior_predict()
    """
    idata: IDPredict

@dataclass
class LogLikResult(RListVectorExtension):
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

@dataclass
class PosteriorLinpredResult(RListVectorExtension):
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

@dataclass
class FormulaResult(RListVectorExtension):
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
    dict: Dict

    @classmethod
    def _formula_parse(cls, r: Union[str, robjects.ListVector]) -> 'FormulaResult':
        from .helpers.conversion import r_to_py
        if isinstance(r, str):
            r_fun = cast(Callable, robjects.r('brms::bf'))
            r = cast(robjects.ListVector, r_fun(r))
        return cls(r=r, dict=cast(dict, r_to_py(r)))


    def __add__(self, other):
        if isinstance(other, (robjects.ListVector, str)):
            other = FormulaResult._formula_parse(other)

        if not isinstance(other, FormulaResult):
            raise ArithmeticError("When adding values to formula, they must be FormulaResult or parseable to FormulaResult")
        
        plus = cast(Callable, robjects.r('function (a, b) a + b'))
        combo = plus(self.r, other.r)

        return FormulaResult._formula_parse(combo)





def _indent_block(text: str, prefix: str = "  ") -> str:
    return "\n".join(prefix + line for line in str(text).splitlines())

@dataclass
class SummaryResult(RListVectorExtension):
    formula: str
    data_name: str
    group: Union[str, List[str]]
    nobs: int
    prior: pd.DataFrame
    algorithm: str
    sampler: str
    total_ndraws: int
    chains: float
    iter: float
    warmup: float
    thin: float
    has_rhat: bool
    fixed: pd.DataFrame
    spec_pars: pd.DataFrame
    cor_pars: pd.DataFrame
    rescor_pars: Optional[pd.DataFrame] = None
    ngrps: Optional[Dict[str, int]] = None
    autocor: Optional[dict] = None
    random: Optional[Dict[str, pd.DataFrame]] = None
    _str: Optional[str] = None

    def __str__(self) -> str:
        if self._str:
            return self._str
        return "MISSING REPR"

    def __repr__(self) -> str:
        # For interactive use, repr == pretty summary
        return self.__str__()
    

@dataclass
class LooResult(RListVectorExtension):
    estimates: pd.DataFrame
    pointwise: pd.DataFrame
    diagnostics: pd.DataFrame
    psis_object: Optional[Any]
    elpd_loo: float
    p_loo: float
    looic: float
    se_elpd_loo: float
    se_p_loo: float
    se_looic: float
    
    def __repr__(self) -> str:
        """Pretty print LOO-CV results."""
        lines = []
        lines.append("LOO-CV Results")
        lines.append("=" * 50)
        lines.append(f"ELPD LOO:  {self.elpd_loo:>10.2f}  (SE: {self.se_elpd_loo:.2f})")
        lines.append(f"p_loo:     {self.p_loo:>10.2f}  (SE: {self.se_p_loo:.2f})")
        lines.append(f"LOOIC:     {self.looic:>10.2f}  (SE: {self.se_looic:.2f})")
        lines.append("")
        
        # Show Pareto k diagnostic summary if available
        if isinstance(self.diagnostics, pd.DataFrame) and not self.diagnostics.empty:
            if 'pareto_k' in self.diagnostics.columns:
                k_vals = self.diagnostics['pareto_k']
                lines.append("Pareto k Diagnostics:")
                lines.append(f"  Good (k < 0.5):       {(k_vals < 0.5).sum():>5} observations")
                lines.append(f"  OK (0.5 ≤ k < 0.7):   {((k_vals >= 0.5) & (k_vals < 0.7)).sum():>5} observations")
                lines.append(f"  Bad (0.7 ≤ k < 1):    {((k_vals >= 0.7) & (k_vals < 1)).sum():>5} observations")
                lines.append(f"  Very bad (k ≥ 1):     {(k_vals >= 1).sum():>5} observations")
                
        return "\n".join(lines)
    

@dataclass
class LooCompareResult(RListVectorExtension):
    table: pd.DataFrame
    criterion: str
    r: robjects.ListVector

    def __repr__(self) -> str:
        header = f"Model comparison ({self.criterion.upper()})"
        lines = [header, "=" * len(header)]
        lines.append(self.table.to_string())
        return "\n".join(lines)
