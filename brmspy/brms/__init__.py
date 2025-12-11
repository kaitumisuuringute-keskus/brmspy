import os
import sys
from contextlib import contextmanager
from types import ModuleType
from typing import TYPE_CHECKING, cast

from brmspy.session.manage import manage
from brmspy.session.session import _INTERNAL_ATTRS, RModuleSession

# -------------------------------------------------------------------
# Typing: describe the brms module surface for static analysis
# -------------------------------------------------------------------
if TYPE_CHECKING:
    # For type checkers / IDE only – can point to the real brms module
    import brmspy.brms._brms_module as _brms_module
    from brmspy.brms._brms_module import *
    from brmspy.brms._brms_module import _runtime

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

    # 3) Wrap it in RModuleSession so all calls go to the worker.
    _module_path = "brmspy.brms"

    brms = cast(
        BrmsModule,
        RModuleSession(module=_brms_module, module_path=_module_path),
    )
    _is_main_process = True
else:
    # WORKER PROCESS
    #
    # Here we *do not* install the stub – worker must see real rpy2.
    # BRMSPY_WORKER=1 should be set in the worker's env before import.
    import brmspy.brms._brms_module as brms

    _is_main_process = False


@contextmanager
def environment():
    if not isinstance(brms, RModuleSession):
        raise Exception("Can't manage environment from WITHIN the environment!")
    # 1. shut down current session
    # 2. yield, let the user install/uninstall packages. give them Ctx object with safe 3-4 functions
    yield
    # 3. do we need cleanup? since we are on a fresh R session, we could just 'continue'


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
]


# Re-export

_this_mod = sys.modules[__name__]

for name in __all__:
    if name in ["manage", "_is_main_process"]:
        continue
    setattr(_this_mod, name, getattr(brms, name))

if _is_main_process:
    for name in _INTERNAL_ATTRS:
        setattr(_this_mod, name, getattr(brms, name))
