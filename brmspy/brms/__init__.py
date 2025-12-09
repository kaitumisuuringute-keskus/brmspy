import os
import sys
from types import ModuleType
from typing import TYPE_CHECKING, cast

from brmspy.session.module_session import RModuleSession

import os
import sys
from types import ModuleType

# -------------------------------------------------------------------
# Typing: describe the brms module surface for static analysis
# -------------------------------------------------------------------
if TYPE_CHECKING:
    # For type checkers / IDE only – can point to the real brms module
    import brmspy._brms_module as _brms_module
    from brmspy._brms_module import *
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
    #install_rpy2_stub()

    # 2) Import the heavy brms module; it will see stubbed rpy2 in main.
    import brmspy._brms_module as _brms_module

    # 3) Wrap it in RModuleSession so all calls go to the worker.
    _module_path = "brmspy.brms"

    brms = cast(
        BrmsModule,
        RModuleSession(
            module=_brms_module,
            module_path=_module_path,
            runtime_conf={
                # TODO: pass R_HOME, startup scripts, etc. here if needed
            },
        ),
    )
else:
    # WORKER PROCESS
    #
    # Here we *do not* install the stub – worker must see real rpy2.
    # BRMSPY_WORKER=1 should be set in the worker's env before import.
    import brmspy._brms_module as brms


__all__ = [

    # R env
    'install_brms', 'install_runtime', 'get_brms_version',  'deactivate_runtime', 'activate_runtime',
    'find_local_runtime', 'get_active_runtime',

    'install_rpackage',

    # IO
    'get_brms_data', 'save_rds', 'read_rds_raw', 'read_rds_fit', 'get_data',

    # brm
    'fit', 'brm',

    # formula
    'formula', 'bf', 'set_mecor', 'set_rescor', 'set_nl',
    'lf', 'nlf', 'acformula',

    # priors
    'prior', 'get_prior', 'default_prior',

    # prediction
    "posterior_predict", "posterior_epred", "posterior_linpred", "log_lik",

    # diagnosis
    'summary', 'fixef', 'ranef', 'posterior_summary', 'prior_summary', 'validate_newdata',
    
    # generic
    'call',

    # families
    "brmsfamily", "family", "student", "bernoulli", "beta_binomial", "negbinomial",
    "negbinomial2", "geometric", "discrete_weibull", "com_poisson", "lognormal",
    "shifted_lognormal", "skew_normal", "exponential", "weibull", "frechet",
    "gen_extreme_value", "exgaussian", "wiener", "Beta", "xbeta", "dirichlet",
    "dirichlet2", "logistic_normal", "von_mises", "asym_laplace",
    "zero_inflated_asym_laplace", "cox", "hurdle_poisson", "hurdle_negbinomial",
    "hurdle_gamma", "hurdle_lognormal", "hurdle_cumulative", "zero_inflated_beta",
    "zero_one_inflated_beta", "zero_inflated_poisson", "zero_inflated_negbinomial",
    "zero_inflated_binomial", "zero_inflated_beta_binomial", "categorical",
    "multinomial", "dirichlet_multinomial", "cumulative", "sratio", "cratio", "acat",
    "gaussian", "poisson", "binomial", "Gamma", "inverse_gaussian",

    # types
    'FitResult', 'FormulaResult', 'PosteriorEpredResult', 'PosteriorPredictResult',
    'PosteriorLinpredResult', 'LogLikResult', 'LooResult', 'LooCompareResult', 'GenericResult', 'RListVectorExtension',

    'IDLinpred',
    'IDEpred',
    'IDFit',
    'IDLogLik',
    'IDPredict',
    'PriorSpec',

    # stan
    'make_stancode',

    '_formula_add'
]

# Re-export

_this_mod = sys.modules[__name__]

for name in __all__:
    setattr(_this_mod, name, getattr(brms, name))
