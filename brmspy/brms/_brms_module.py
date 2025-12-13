# ruff: noqa
"""
Main brms module with Pythonic API.
"""

from brmspy.helpers.log import log
from brmspy import _runtime
from brmspy.helpers.log import log_warning
from brmspy._runtime._state import get_brms as _get_brms
from brmspy.types.brms_results import (
    FitResult,
    GenericResult,
    LogLikResult,
    LooResult,
    LooCompareResult,
    PosteriorEpredResult,
    PosteriorLinpredResult,
    PosteriorPredictResult,
    RListVectorExtension,
    IDLinpred,
    IDEpred,
    IDFit,
    IDLogLik,
    IDPredict,
    PriorSpec,
)
from brmspy.types.formula_dsl import FormulaConstruct, FormulaPart
from brmspy._brms_functions.io import (
    get_brms_data,
    read_rds_fit,
    read_rds_raw,
    save_rds,
    get_data,
)
from brmspy._brms_functions.prior import prior, get_prior, default_prior
from brmspy._brms_functions.brm import brm
from brmspy._brms_functions.brm import brm as fit

from brmspy._brms_functions.diagnostics import (
    summary,
    fixef,
    ranef,
    posterior_summary,
    prior_summary,
    validate_newdata,
)
from brmspy._brms_functions.generic import call
from brmspy._brms_functions.formula import (
    bf,
    lf,
    nlf,
    acformula,
    set_rescor,
    set_mecor,
    set_nl,
)
from brmspy._brms_functions.formula import bf as formula
from brmspy._brms_functions.prediction import (
    posterior_epred,
    posterior_linpred,
    posterior_predict,
    log_lik,
)
from brmspy._brms_functions.stan import make_stancode
from brmspy._brms_functions.families import (
    brmsfamily,
    family,
    student,
    bernoulli,
    beta_binomial,
    negbinomial,
    negbinomial2,
    geometric,
    discrete_weibull,
    com_poisson,
    lognormal,
    shifted_lognormal,
    skew_normal,
    exponential,
    weibull,
    frechet,
    gen_extreme_value,
    exgaussian,
    wiener,
    Beta,
    xbeta,
    dirichlet,
    dirichlet2,
    logistic_normal,
    von_mises,
    asym_laplace,
    zero_inflated_asym_laplace,
    cox,
    hurdle_poisson,
    hurdle_negbinomial,
    hurdle_gamma,
    hurdle_lognormal,
    hurdle_cumulative,
    zero_inflated_beta,
    zero_one_inflated_beta,
    zero_inflated_poisson,
    zero_inflated_negbinomial,
    zero_inflated_binomial,
    zero_inflated_beta_binomial,
    categorical,
    multinomial,
    dirichlet_multinomial,
    cumulative,
    sratio,
    cratio,
    acat,
    gaussian,
    poisson,
    binomial,
    Gamma,
    inverse_gaussian,
)

from brmspy._runtime import (
    get_brms_version,
    find_local_runtime,
    get_active_runtime,
    status,
)
import brmspy._runtime as _runtime


# Auto-load last runtime on import
import os

if os.environ.get("BRMSPY_WORKER") == "1" and os.environ.get("BRMSPY_AUTOLOAD") == "1":
    log("Running autoload!")
    _runtime._autoload()


# R imports must NOT be done lazily!
# Lazy imports with rpy2 within tqdm loops for example WILL cause segfaults!
# This can lead to wild and unexpected behaviour, hence we do R imports when brms.py is imported

try:
    if (
        os.environ.get("BRMSPY_WORKER") == "1"
        and os.environ.get("BRMSPY_AUTOLOAD") == "1"
    ):
        _get_brms()
except ImportError:
    log_warning(
        "brmspy: brms and other required libraries are not installed. Please call brmspy.install_brms()"
    )
