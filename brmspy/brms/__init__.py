"""
User-facing brms API.

Import this module to call `brms` functions from Python (for example `brm`,
`prior`, `posterior_predict`, etc.). brmspy runs these calls through an isolated
runtime so that R-side instability is less likely to take down your Python
process.

Use `brms.manage()` to install brms / CmdStan, and to work with multiple
isolated environments.

Examples
--------
```python
from brmspy import brms
with brms.manage(environment_name="default") as ctx:
    ctx.install_brms(use_prebuilt=True)
```
"""

import os
import sys
from types import ModuleType
from typing import TYPE_CHECKING, cast

from brmspy._session.session import _INTERNAL_ATTRS, RModuleSession

# -------------------------------------------------------------------
# Typing: describe the brms module surface for static analysis
# -------------------------------------------------------------------
if TYPE_CHECKING:
    # For type checkers / IDE only – can point to the real brms module
    import brmspy.brms._brms_module as _brms_module
    from brmspy.brms._brms_module import *
    from brmspy.brms._brms_module import _runtime

    from contextlib import AbstractContextManager

    from brmspy.brms._build_module import BuildModule
    from brmspy.brms._manage_module import ManageModule
    from brmspy.types.session import EnvironmentConfig

    # Stubs for IDEs: these are attached dynamically in the main process.
    def manage(
        *,
        environment_config: EnvironmentConfig | dict[str, str] | None = None,
        environment_name: str | None = None,
    ) -> AbstractContextManager[ManageModule]: ...

    def _build(
        *,
        environment_config: EnvironmentConfig | dict[str, str] | None = None,
        environment_name: str | None = None,
    ) -> AbstractContextManager[BuildModule]: ...

    def environment_activate(name: str): ...
    def environment_exists(name: str) -> bool: ...

    BrmsModule = _brms_module
else:
    # At runtime, just treat it as a generic module
    BrmsModule = ModuleType  # type: ignore[assignment]

# -------------------------------------------------------------------
# Runtime wiring: proxy in main, real module in worker
# -------------------------------------------------------------------

if os.environ.get("BRMSPY_WORKER") != "1":
    # MAIN PROCESS
    #
    # 1) Ensure rpy2 is stubbed before importing brmspy.brms,
    #    so any top-level rpy2 imports in that module are safe.
    # install_rpy2_stub()

    # 2) Import the heavy brms module; it will see stubbed rpy2 in main.
    import brmspy.brms._brms_module as _brms_module

    # 3) Import surface classes (must be safe to import in main).
    from brmspy.brms._build_module import BuildModule
    from brmspy.brms._manage_module import ManageModule

    # 4) Wrap brms module in RModuleSession so all calls go to the worker.
    _module_path = "brmspy.brms"
    _sess = RModuleSession(module=_brms_module, module_path=_module_path)

    # 5) Attach context-managed surfaces (dynamic attributes)
    setattr(
        _sess,
        "manage",
        _sess.add_contextmanager(
            surface_class=ManageModule,
            surface_class_path="brmspy.brms._manage_module.ManageModule",
        ),
    )
    setattr(
        _sess,
        "_build",
        _sess.add_contextmanager(
            surface_class=BuildModule,
            surface_class_path="brmspy.brms._build_module.BuildModule",
        ),
    )

    brms = cast(BrmsModule, _sess)
    _is_main_process = True

    # Sanity check that rpy2.robjects wasnt imported
    banned = (
        "rpy2.robjects",
        "rpy2.robjects.packages",
        "rpy2.robjects.vectors",
    )
    present = [m for m in banned if m in sys.modules]
    if present:
        raise RuntimeError(
            "Sanity check failed: rpy2.robjects was imported on the main process. "
            f"Present: {present}. This should only happen inside the worker."
        )
else:
    # WORKER PROCESS
    import brmspy.brms._brms_module as brms

    _is_main_process = False


__all__ = [
    # R env
    "get_brms_version",
    "find_local_runtime",
    "get_active_runtime",
    "manage",
    "_is_main_process",
    # IO
    "get_brms_data",
    "save_rds",
    "read_rds_raw",
    "read_rds_fit",
    "get_data",
    # brm
    "fit",
    "brm",
    # formula
    "formula",
    "bf",
    "set_mecor",
    "set_rescor",
    "set_nl",
    "lf",
    "nlf",
    "acformula",
    # priors
    "prior",
    "get_prior",
    "default_prior",
    # prediction
    "posterior_predict",
    "posterior_epred",
    "posterior_linpred",
    "log_lik",
    # diagnosis
    "summary",
    "fixef",
    "ranef",
    "posterior_summary",
    "prior_summary",
    "validate_newdata",
    # generic
    "call",
    # families
    "brmsfamily",
    "family",
    "student",
    "bernoulli",
    "beta_binomial",
    "negbinomial",
    "negbinomial2",
    "geometric",
    "discrete_weibull",
    "com_poisson",
    "lognormal",
    "shifted_lognormal",
    "skew_normal",
    "exponential",
    "weibull",
    "frechet",
    "gen_extreme_value",
    "exgaussian",
    "wiener",
    "Beta",
    "xbeta",
    "dirichlet",
    "dirichlet2",
    "logistic_normal",
    "von_mises",
    "asym_laplace",
    "zero_inflated_asym_laplace",
    "cox",
    "hurdle_poisson",
    "hurdle_negbinomial",
    "hurdle_gamma",
    "hurdle_lognormal",
    "hurdle_cumulative",
    "zero_inflated_beta",
    "zero_one_inflated_beta",
    "zero_inflated_poisson",
    "zero_inflated_negbinomial",
    "zero_inflated_binomial",
    "zero_inflated_beta_binomial",
    "categorical",
    "multinomial",
    "dirichlet_multinomial",
    "cumulative",
    "sratio",
    "cratio",
    "acat",
    "gaussian",
    "poisson",
    "binomial",
    "Gamma",
    "inverse_gaussian",
    # types
    "FitResult",
    "FormulaConstruct",
    "FormulaPart",
    "PosteriorEpredResult",
    "PosteriorPredictResult",
    "PosteriorLinpredResult",
    "LogLikResult",
    "LooResult",
    "LooCompareResult",
    "GenericResult",
    "RListVectorExtension",
    "IDLinpred",
    "IDEpred",
    "IDFit",
    "IDLogLik",
    "IDPredict",
    "PriorSpec",
    # stan
    "make_stancode",
    # misc private
    "_runtime",
    "status",
    "manage",
    "_build",
    "environment_exists",
    "environment_activate",
]


# Re-export

_this_mod = sys.modules[__name__]

for name in __all__:
    if hasattr(brms, name):
        setattr(_this_mod, name, getattr(brms, name))

if _is_main_process:
    for name in _INTERNAL_ATTRS:
        setattr(_this_mod, name, getattr(brms, name))
