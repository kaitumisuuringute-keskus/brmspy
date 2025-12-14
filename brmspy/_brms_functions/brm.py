"""
Model fitting wrappers.

This module contains the `brms::brm()` wrapper used by `brmspy.brms.fit()` /
`brmspy.brms.brm()`.

Notes
-----
This code executes inside the worker process (the process that hosts the embedded
R session).
"""

from collections.abc import Callable, Sequence
from typing import Any, cast

import pandas as pd
from rpy2.rinterface import ListSexpVector
from rpy2.rinterface_lib import openrlib

from brmspy.helpers.log import log, log_warning
from brmspy.types.session import SexpWrapper

from ..helpers._rpy2._conversion import brmsfit_to_idata, kwargs_r, py_to_r
from ..helpers._rpy2._priors import _build_priors
from ..types.brms_results import FitResult, IDFit, PriorSpec, ProxyListSexpVector
from ..types.formula_dsl import FormulaConstruct
from .formula import _execute_formula, bf

_formula_fn = bf


_WARNING_CORES = (
    "`cores <= 1` can be unstable in embedded R sessions and may crash the worker "
    "process. Prefer `cores >= 2`."
)


def _warn_cores(cores: int | None):
    if cores is None or cores <= 1:
        log_warning(_WARNING_CORES)


def brm(
    formula: FormulaConstruct | ProxyListSexpVector | str,
    data: dict | pd.DataFrame,
    priors: Sequence[PriorSpec] | None = None,
    family: str | ListSexpVector | None = "gaussian",
    sample_prior: str = "no",
    sample: bool = True,
    backend: str = "cmdstanr",
    formula_args: dict | None = None,
    cores: int | None = 2,
    **brm_args,
) -> FitResult:
    """
    Fit a Bayesian regression model with brms.

    This is a thin wrapper around R ``brms::brm()`` that returns a structured
    `FitResult` (including an ArviZ `InferenceData`).

    Parameters
    ----------
    formula : str or FormulaConstruct
        Model formula. Accepts a plain brms formula string (e.g. ``"y ~ x + (1|g)"``)
        or a composed formula created via `brmspy.brms.bf()` / `brmspy.brms.lf()`
        (typically imported as ``from brmspy.brms import bf, lf``).
    data : dict or pandas.DataFrame
        Model data.
    priors : Sequence[PriorSpec] or None, default=None
        Optional prior specifications created via `brmspy.brms.prior()`.
    family : str or rpy2.rinterface.ListSexpVector or None, default="gaussian"
        brms family specification (e.g. ``"gaussian"``, ``"poisson"``).
    sample_prior : str, default="no"
        Passed to brms. Common values: ``"no"``, ``"yes"``, ``"only"``.
    sample : bool, default=True
        If ``False``, compile the model without sampling (brms ``empty=TRUE``).
    backend : str, default="cmdstanr"
        Stan backend. Common values: ``"cmdstanr"`` or ``"rstan"``.
    formula_args : dict or None, default=None
        Reserved for future use. Currently ignored.
    cores : int or None, default=2
        Number of cores for brms/cmdstanr.
    **brm_args
        Additional keyword arguments passed to R ``brms::brm()`` (e.g. ``chains``,
        ``iter``, ``warmup``, ``seed``).

    Returns
    -------
    FitResult
        Result object with `idata` (ArviZ `InferenceData`) and an underlying R handle.

    See Also
    --------
    brms::brm : [R documentation](https://paulbuerkner.com/brms/reference/brm.html)

    Warnings
    --------
    Using ``cores <= 1`` can be unstable in embedded R sessions and may crash the
    worker process. Prefer ``cores >= 2``.

    Examples
    --------
    ```python
    from brmspy import brms

    fit = brms.brm("y ~ x + (1|g)", data=df, family="gaussian", chains=4, cores=4)

    fit.idata.posterior
    ```
    """
    import rpy2.robjects as ro
    import rpy2.robjects.packages as packages

    fun_brm = cast(Callable, ro.r("brms::brm"))

    if backend == "cmdstanr":
        try:
            cmdstanr = packages.importr("cmdstanr")
        except:
            cmdstanr = None
        if cmdstanr is None:
            raise RuntimeError(
                "cmdstanr backend is not installed! Please run install_brms(install_cmdstanr=True)"
            )

    if backend == "rstan":
        try:
            rstan = packages.importr("rstan")
        except:
            rstan = None
        if rstan is None:
            raise RuntimeError(
                "rstan backend is not installed! Please run install_brms(install_rstan=True)"
            )

    # Formula checks. These should never be reached in the first place
    # if they are, the library is calling brm() from main directly without remote call
    assert not isinstance(formula, SexpWrapper)
    assert formula is not None
    formula_obj = _execute_formula(formula)

    # Convert data to R format
    data_r = py_to_r(data)

    # Setup priors
    brms_prior = _build_priors(priors)

    # Prepare brm() arguments
    brm_kwargs: dict[str, Any] = {
        "formula": formula_obj,
        "data": data_r,
        "family": family,
        "sample_prior": sample_prior,
        "backend": backend,
        "cores": cores,
    }

    # Add priors if specified
    if len(brms_prior) > 0:
        brm_kwargs["prior"] = brms_prior

    # Add user-specified arguments
    brm_kwargs.update(brm_args)

    brm_kwargs = kwargs_r(brm_kwargs)

    # Set empty=TRUE if not sampling
    if not sample:
        brm_kwargs["empty"] = True
        log("Creating empty r object (no sampling)...")
    else:
        log(f"Fitting model with brms (backend: {backend})...")

    # Call brms::brm() with all arguments
    fit = fun_brm(**brm_kwargs)

    log("Fit done!")

    # Handle return type conversion
    if not sample:
        return FitResult(idata=IDFit(), r=fit)

    idata = brmsfit_to_idata(fit)
    return FitResult(idata=idata, r=fit)
