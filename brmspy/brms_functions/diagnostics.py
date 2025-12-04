from dataclasses import dataclass
from types import UnionType
from typing import Any, Callable, Dict, Optional, TypedDict, Union, cast, get_origin
import typing
import pandas as pd
from rpy2.robjects import default_converter, pandas2ri
import pandas as pd

from brmspy.helpers.log import log_warning
from brmspy.helpers.robject_iter import iterate_robject_to_dataclass
from ..helpers.conversion import (
    kwargs_r,
    py_to_r,
    r_to_py
)
from ..types import (
    FitResult
)
import rpy2.robjects as ro


def _indent_block(text: str, prefix: str = "  ") -> str:
    return "\n".join(prefix + line for line in str(text).splitlines())


@dataclass
class Summary:
    formula: str
    data_name: str
    group: str
    nobs: int
    ngrps: Dict[str, int]
    autocor: Optional[dict]
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
    random: dict

    def __str__(self) -> str:
        lines = []

        # Header (roughly analogous to brms::summary header)
        lines.append("Summary of brmsfit (Python)")
        lines.append("")
        lines.append(f"Formula: {self.formula}")
        lines.append(
            f"   Data: {self.data_name} (Number of observations: {self.nobs})"
        )
        lines.append(
            "  Draws: "
            f"{self.chains:g} chains, each with iter = {self.iter:g}; "
            f"warmup = {self.warmup:g}; thin = {self.thin:g};"
        )
        lines.append(
            f"         total post-warmup draws = {self.total_ndraws}"
        )

        # Group-level info
        if self.ngrps:
            lines.append("")
            lines.append("Group-Level Effects:")
            grp_parts = [f"{name} ({n})" for name, n in self.ngrps.items()]
            lines.append("  Groups: " + ", ".join(grp_parts))

            if self.random:
                # brms usually has a list of data.frames here; we try to mirror that
                if isinstance(self.random, dict):
                    for gname, val in self.random.items():
                        lines.append(f" ~{gname}")
                        if isinstance(val, pd.DataFrame):
                            lines.append(_indent_block(val.to_string(), "  "))
                        else:
                            lines.append(_indent_block(str(val), "  "))
                else:
                    lines.append(_indent_block(str(self.random), "  "))

        # Population-level effects
        if isinstance(self.fixed, pd.DataFrame) and not self.fixed.empty:
            lines.append("")
            lines.append("Population-Level Effects:")
            lines.append(_indent_block(self.fixed.to_string(), "  "))

        # Family-specific parameters (spec_pars)
        if isinstance(self.spec_pars, pd.DataFrame) and not self.spec_pars.empty:
            lines.append("")
            lines.append("Family Specific Parameters:")
            lines.append(_indent_block(self.spec_pars.to_string(), "  "))

        # Correlation parameters (cor_pars)
        if isinstance(self.cor_pars, pd.DataFrame) and not self.cor_pars.empty:
            lines.append("")
            lines.append("Correlation Parameters:")
            lines.append(_indent_block(self.cor_pars.to_string(), "  "))

        # Prior info (optional but often useful)
        if isinstance(self.prior, pd.DataFrame) and not self.prior.empty:
            lines.append("")
            lines.append("Prior:")
            lines.append(_indent_block(self.prior.to_string(), "  "))

        # Extra diagnostics / meta info
        lines.append("")
        lines.append("Algorithm & Diagnostics:")
        lines.append(f"  Algorithm: {self.algorithm}")
        lines.append(f"  Sampler:   {self.sampler}")
        lines.append(f"  Rhat:      {'reported' if self.has_rhat else 'not reported'}")
        if self.autocor is not None:
            lines.append(f"  Autocor:   {self.autocor}")
        else:
            lines.append("  Autocor:   None")

        return "\n".join(lines)

    def __repr__(self) -> str:
        # For interactive use, repr == pretty summary
        return self.__str__()


def summary(model: FitResult, **kwargs) -> Summary:
    """
    Generate summary statistics for fitted model.
    
    [BRMS documentation and parameters](https://paulbuerkner.com/brms/reference/summary.brmsfit.html)
    
    Parameters
    ----------
    model : FitResult
        Fitted model from fit()
    **kwargs
        Additional arguments to summary(), e.g., probs=c(0.025, 0.975)
    
    Returns
    -------
    
    See Also
    --------
    brms::summary.brmsfit : R documentation
        https://paulbuerkner.com/brms/reference/summary.brmsfit.html
    
    Examples
    --------
    ```python
    from brmspy import brms
    
    model = brms.fit("y ~ x", data=data, chains=4)
    df_fixed_effects = brms.summary(model)['fixed']
    print(df_fixed_effects)
    ```
    """
    
    kwargs = kwargs_r(kwargs)
    r_summary = cast(Callable, ro.r('summary'))
    summary_r = r_summary(model.r, **kwargs)


    _default_get_r = lambda param: f'function(x) x${param}'
    _get_methods_r: Dict[str, Callable[[str], str]] = {
        # Extract a clean formula string: "y ~ x1 + x2 + ..."
        "formula": lambda param: (
            "function(x) { paste(deparse(x$formula$formula), collapse = ' ') }"
        ),
    }

    names = summary_r.names
    get = lambda param: r_to_py(cast(Callable, ro.r(_get_methods_r.get(param, _default_get_r)(param)))(summary_r))
    out = iterate_robject_to_dataclass(names=names, get=get, target_dataclass=Summary)

    return cast(Summary, out)


def fixef(
  object: Union[FitResult, ro.ListVector],
  summary = True,
  robust = False,
  probs = (0.025, 0.975),
  pars = None,
  **kwargs
) -> pd.DataFrame:
    obj_r = py_to_r(object)
    kwargs = kwargs_r(kwargs)

    return pd.DataFrame({})