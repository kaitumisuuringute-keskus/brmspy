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
    from contextlib import AbstractContextManager

    import brmspy.brms._brms_module as _brms_module
    from brmspy.brms._brms_module import *
    from brmspy.brms._brms_module import _runtime
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

    # 5) Intercept backend="nutpie" in the main process.
    # Capture the original proxy wrapper now, before we overwrite "brm" in
    # _func_cache — the non-nutpie fallback path must call this, not _sess.brm
    # (which would resolve to _brm_main_process and recurse infinitely).
    _brm_proxy = _sess.brm

    #    make_stancode / make_standata are called through the proxy (worker),
    #    then nutpie compiles and samples right here so that its progress bar
    #    and Jupyter widgets work correctly.
    def _brm_nutpie_main(
        formula,
        data,
        priors=None,
        family="gaussian",
        sample_prior="no",
        formula_args=None,
        cores=None,
        nutpie_config=None,
        brm_args=None,
    ):
        try:
            import nutpie
        except ImportError as exc:
            raise ImportError(
                "nutpie is not installed.  Install it with:\n\n"
                '    pip install "nutpie[stan]"\n\n'
                "or via conda:\n\n"
                "    conda install -c conda-forge nutpie"
            ) from exc

        from typing import cast

        import pandas as pd

        from brmspy.types.brms_results import FitResult, IDBrm, NutpieConfig

        if nutpie_config is None:
            nutpie_config = NutpieConfig()

        # Normalise formula_args into the formula string before shipping to worker
        if formula_args and isinstance(formula, str):
            # bf() must run in the worker; pass formula_args as part of the
            # make_stancode / make_standata calls instead
            pass

        data_df = pd.DataFrame(data) if isinstance(data, dict) else data
        family_str = family if isinstance(family, str) else "gaussian"

        # Step 1 + 2 – run in worker via proxy
        stan_code = _sess.make_stancode(
            formula=formula,
            data=data_df,
            priors=priors,
            family=family_str,
            sample_prior=sample_prior,
            formula_args=formula_args,
        )
        stan_data = _sess.make_standata(
            formula=formula,
            data=data_df,
            priors=priors,
            family=family_str,
            sample_prior=sample_prior,
            formula_args=formula_args,
            coerce_types=True,
        )

        # Step 3 – compile (main process)
        compiled = nutpie.compile_stan_model(code=stan_code)

        # Step 4 – attach data (main process)
        compiled = compiled.with_data(**stan_data)

        # Step 5 – sample (main process, progress bar works here)
        sample_kwargs = nutpie_config.to_sample_kwargs()

        # Map common brm()/Stan arguments onto nutpie as fallbacks:
        # only applied when the NutpieConfig field was left as None (i.e. not
        # explicitly configured by the user via nutpie_config).
        if brm_args:
            # chains= is a first-class brm() arg, but also in brm_args if passed
            # as a kwarg; NutpieConfig.chains has a non-None default (6) so we
            # only override when the user passed chains= explicitly to brm().
            if "chains" in brm_args:
                sample_kwargs["chains"] = brm_args["chains"]

            # seed= maps directly
            if "seed" in brm_args and nutpie_config.seed is None:
                sample_kwargs["seed"] = brm_args["seed"]

            # warmup= maps to tune=
            if "warmup" in brm_args and nutpie_config.tune is None:
                sample_kwargs["tune"] = brm_args["warmup"]

            # iter= in brms means total draws including warmup; nutpie's draws=
            # means post-warmup draws, so: draws = iter - warmup.
            if "iter" in brm_args and nutpie_config.draws is None:
                warmup = brm_args.get("warmup", sample_kwargs.get("tune", 300))
                sample_kwargs["draws"] = brm_args["iter"] - warmup

        if cores is not None:
            sample_kwargs["cores"] = cores
        trace = nutpie.sample(compiled, **sample_kwargs)

        return FitResult(idata=cast(IDBrm, trace), r=None)

    def _brm_main_process(
        formula,
        data,
        priors=None,
        family="gaussian",
        sample_prior="no",
        sample=True,
        backend="cmdstanr",
        formula_args=None,
        cores=2,
        nutpie_config=None,
        *,
        return_idata=True,
        **brm_args,
    ):
        if backend == "nutpie":
            return _brm_nutpie_main(
                formula=formula,
                data=data,
                priors=priors,
                family=family,
                sample_prior=sample_prior,
                formula_args=formula_args,
                cores=cores,
                nutpie_config=nutpie_config,
                brm_args=brm_args,
            )
        # All other backends go through the worker as normal
        return _brm_proxy(
            formula=formula,
            data=data,
            priors=priors,
            family=family,
            sample_prior=sample_prior,
            sample=sample,
            backend=backend,
            formula_args=formula_args,
            cores=cores,
            return_idata=return_idata,
            **brm_args,
        )

    # Copy metadata from the worker-side brm for IDE/doc purposes
    _brm_main_process.__doc__ = _brms_module.brm.__doc__
    _brm_main_process.__name__ = "brm"
    _brm_main_process.__wrapped__ = _brms_module.brm  # type: ignore[attr-defined]

    # Register directly in _func_cache so __getattribute__ finds it at step 2
    # (before it falls through to the module and creates a proxy wrapper that
    # would send nutpie_config to the worker).
    func_cache = ModuleType.__getattribute__(_sess, "_func_cache")
    func_cache["brm"] = _brm_main_process
    func_cache["fit"] = _brm_main_process

    # 6) Attach context-managed surfaces (dynamic attributes)
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
    "posterior",
    "observed_data",
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
    "FormulaConstruct",
    "FormulaPart",
    "FitResult",
    "IDResult",
    # "LooResult",
    # "LooCompareResult",
    "IDPosterior",
    "IDPosteriorPredictive",
    "IDPredictions",
    "IDLogLikelihoodInsample",
    "IDLogLikelihoodOutsample",
    "IDObservedData",
    "IDConstantData",
    "IDPredictionsConstantData",
    "RListVectorExtension",
    "IDBrm",
    "PriorSpec",
    # stan
    "make_stancode",
    "make_standata",
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
