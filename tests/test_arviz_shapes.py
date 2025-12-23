"""
Spec-driven tests for ArviZ InferenceData shapes/groups produced by brmspy.

These tests intentionally encode the "markdown spec" contract:

- canonical observation dim name: obs_id
- deterministic shapes: (chain=C, draw=D, obs_id=N or M)
- group names differ between in-sample and out-of-sample prediction calls
- variable names must match exact spec: count/Base, *_mean, *_linpred
- obs_id must be carried over from pandas index for both training data and newdata

If implementation drifts from the spec, these tests should fail.
"""

from __future__ import annotations

import arviz as az
import numpy as np
import pandas as pd
import pytest
import xarray as xr

from brmspy.types.brms_results import (
    FitResult,
    IDBrm,
    IDLogLikelihoodInsample,
    IDObservedData,
    IDPosterior,
    IDPosteriorPredictive,
    IDPredictions,
)


# -------------------------
# Shared spec constants
# -------------------------

C = 2
ITER = 200
WARMUP = 100
D = ITER - WARMUP

SEED = 1234

# out-of-sample row count
M = 25


# -------------------------
# Helper assertions
# -------------------------


def _groups(idata: az.InferenceData) -> set[str]:
    return set(idata.groups())


def _assert_groups_exact(idata: az.InferenceData, expected: set[str]) -> None:
    assert (
        _groups(idata) == expected
    ), f"groups={idata.groups()!r}, expected={sorted(expected)!r}"


def _assert_da_dims_and_shape(
    da: xr.DataArray, *, dims: tuple[str, ...], shape: tuple[int, ...]
) -> None:
    assert tuple(da.dims) == dims, f"dims={da.dims!r}, expected={dims!r}"
    assert tuple(da.shape) == shape, f"shape={da.shape!r}, expected={shape!r}"


def _assert_obs_id_coords_exact(
    da_or_ds: xr.DataArray | xr.Dataset, expected_obs_id: np.ndarray
) -> None:
    actual = np.asarray(da_or_ds.coords["obs_id"].values)
    np.testing.assert_array_equal(actual, expected_obs_id)


def _assert_dataset_has_vars(ds: xr.Dataset, expected_vars: set[str]) -> None:
    missing = expected_vars.difference(set(ds.data_vars))
    assert (
        not missing
    ), f"missing vars {sorted(missing)!r}, got {sorted(ds.data_vars)!r}"


def _assert_dataset_lacks_vars(ds: xr.Dataset, forbidden_vars: set[str]) -> None:
    present = forbidden_vars.intersection(set(ds.data_vars))
    assert not present, f"forbidden vars present {sorted(present)!r}"


def _assert_constant_dataset(
    ds: xr.Dataset,
    *,
    expected_obs_id: np.ndarray,
    must_include: set[str],
    must_exclude: set[str],
    expected_n: int,
) -> None:
    _assert_dataset_has_vars(ds, must_include)
    _assert_dataset_lacks_vars(ds, must_exclude)
    _assert_obs_id_coords_exact(ds, expected_obs_id)

    for v in must_include:
        var = ds[v]
        _assert_da_dims_and_shape(var, dims=("obs_id",), shape=(expected_n,))


# -------------------------
# Data + model fixtures
# -------------------------


@pytest.fixture(scope="module")
def epilepsy_df() -> pd.DataFrame:
    from brmspy import brms

    df = brms.get_brms_data("epilepsy").copy()
    df.index = [f"o{i}" for i in range(len(df))]
    return df


@pytest.fixture(scope="module")
def newdata_df(epilepsy_df: pd.DataFrame) -> pd.DataFrame:
    # sample rows (replace allowed), then force a fresh obs_id sequence on index
    newdata = epilepsy_df.sample(M, replace=True, random_state=SEED).copy()
    newdata.index = [f"pred{i}" for i in range(len(newdata))]
    return newdata


@pytest.fixture(scope="module")
def formula_uni():
    from brmspy.brms import bf, poisson

    return bf("count ~ zAge + zBase * Trt + (1|patient)") + poisson()


@pytest.fixture(scope="module")
def formula_multi():
    from brmspy.brms import bf, poisson, set_rescor

    return (
        (bf("count ~ zAge + zBase * Trt + visit + (1 + visit | patient)") + poisson())
        + (bf("Base ~ zAge + Trt + (1 | patient)") + poisson())
        + set_rescor(False)
    )


@pytest.fixture(scope="module")
def fit_uni(epilepsy_df: pd.DataFrame, formula_uni) -> FitResult:
    from brmspy import brms

    return brms.brm(
        formula_uni,
        data=epilepsy_df,
        chains=C,
        iter=ITER,
        warmup=WARMUP,
        seed=SEED,
        cores=2,
        refresh=0,
        silent=2,
    )


@pytest.fixture(scope="module")
def fit_multi(epilepsy_df: pd.DataFrame, formula_multi) -> FitResult:
    from brmspy import brms

    return brms.brm(
        formula_multi,
        data=epilepsy_df,
        chains=C,
        iter=ITER,
        warmup=WARMUP,
        seed=SEED,
        cores=2,
        refresh=0,
        silent=2,
    )


# ------------------------------------
# brm() idata contract (top-level spec)
# ------------------------------------


@pytest.mark.requires_brms
@pytest.mark.slow
def test_brm_idata_groups_univariate(
    fit_uni: FitResult, epilepsy_df: pd.DataFrame
) -> None:
    idata: IDBrm = fit_uni.idata
    _assert_groups_exact(
        idata,
        {
            "posterior",
            "posterior_predictive",
            "log_likelihood",
            "observed_data",
            "constant_data",
        },
    )

    # observed response exists and is 1D over obs_id
    obs_id = np.asarray(epilepsy_df.index.to_numpy())
    _assert_dataset_has_vars(idata.observed_data, {"count"})
    _assert_da_dims_and_shape(
        idata.observed_data["count"], dims=("obs_id",), shape=(len(epilepsy_df),)
    )
    _assert_obs_id_coords_exact(idata.observed_data, obs_id)

    # in-sample ppc + loglik each (C, D, N)
    _assert_dataset_has_vars(idata.posterior_predictive, {"count"})
    _assert_da_dims_and_shape(
        idata.posterior_predictive["count"],
        dims=("chain", "draw", "obs_id"),
        shape=(C, D, len(epilepsy_df)),
    )
    _assert_obs_id_coords_exact(idata.posterior_predictive["count"], obs_id)

    _assert_dataset_has_vars(idata.log_likelihood, {"count"})
    _assert_da_dims_and_shape(
        idata.log_likelihood["count"],
        dims=("chain", "draw", "obs_id"),
        shape=(C, D, len(epilepsy_df)),
    )
    _assert_obs_id_coords_exact(idata.log_likelihood["count"], obs_id)

    # constant_data: must include predictors/ids (at least), must exclude responses
    _assert_constant_dataset(
        idata.constant_data,
        expected_obs_id=obs_id,
        must_include={"zAge", "zBase", "Trt", "patient"},
        must_exclude={"count"},
        expected_n=len(epilepsy_df),
    )


@pytest.mark.requires_brms
@pytest.mark.slow
def test_brm_idata_groups_multivariate(
    fit_multi: FitResult, epilepsy_df: pd.DataFrame
) -> None:
    idata: IDBrm = fit_multi.idata
    _assert_groups_exact(
        idata,
        {
            "posterior",
            "posterior_predictive",
            "log_likelihood",
            "observed_data",
            "constant_data",
        },
    )

    obs_id = np.asarray(epilepsy_df.index.to_numpy())

    # observed_data: both responses, each (N,)
    _assert_dataset_has_vars(idata.observed_data, {"count", "Base"})
    for resp in ("count", "Base"):
        _assert_da_dims_and_shape(
            idata.observed_data[resp], dims=("obs_id",), shape=(len(epilepsy_df),)
        )
        _assert_obs_id_coords_exact(idata.observed_data[resp], obs_id)

    # posterior_predictive + log_likelihood: each response is (C, D, N)
    _assert_dataset_has_vars(idata.posterior_predictive, {"count", "Base"})
    _assert_dataset_has_vars(idata.log_likelihood, {"count", "Base"})
    for resp in ("count", "Base"):
        _assert_da_dims_and_shape(
            idata.posterior_predictive[resp],
            dims=("chain", "draw", "obs_id"),
            shape=(C, D, len(epilepsy_df)),
        )
        _assert_da_dims_and_shape(
            idata.log_likelihood[resp],
            dims=("chain", "draw", "obs_id"),
            shape=(C, D, len(epilepsy_df)),
        )
        _assert_obs_id_coords_exact(idata.posterior_predictive[resp], obs_id)
        _assert_obs_id_coords_exact(idata.log_likelihood[resp], obs_id)

    # constant_data: must include predictors/ids (at least), must exclude responses
    _assert_constant_dataset(
        idata.constant_data,
        expected_obs_id=obs_id,
        must_include={"zAge", "zBase", "Trt", "visit", "patient"},
        must_exclude={"count", "Base"},
        expected_n=len(epilepsy_df),
    )


# ----------------------------------------
# Univariate prediction function contracts
# ----------------------------------------


@pytest.mark.requires_brms
@pytest.mark.slow
def test_univariate_posterior_groups_and_constant_data(
    fit_uni: FitResult, epilepsy_df: pd.DataFrame
) -> None:
    from brmspy import brms

    res = brms.posterior(fit_uni)
    idata: IDPosterior = res.idata
    _assert_groups_exact(idata, {"posterior", "constant_data"})

    # posterior: parameters only, but must have (chain, draw, ...)
    assert "chain" in idata.posterior.dims
    assert "draw" in idata.posterior.dims
    assert idata.posterior.sizes["chain"] == C
    assert idata.posterior.sizes["draw"] == D

    # constant_data shape and obs_id
    obs_id = np.asarray(epilepsy_df.index.to_numpy())
    _assert_constant_dataset(
        idata.constant_data,
        expected_obs_id=obs_id,
        must_include={"zAge", "zBase", "Trt", "patient"},
        must_exclude={"count"},
        expected_n=len(epilepsy_df),
    )


@pytest.mark.requires_brms
@pytest.mark.slow
def test_univariate_observed_data_groups_and_obs_id(
    fit_uni: FitResult, epilepsy_df: pd.DataFrame
) -> None:
    from brmspy import brms

    res = brms.observed_data(fit_uni)
    idata: IDObservedData = res.idata
    _assert_groups_exact(idata, {"observed_data", "constant_data"})

    obs_id = np.asarray(epilepsy_df.index.to_numpy())
    _assert_dataset_has_vars(idata.observed_data, {"count"})
    _assert_da_dims_and_shape(
        idata.observed_data["count"], dims=("obs_id",), shape=(len(epilepsy_df),)
    )
    _assert_obs_id_coords_exact(idata.observed_data, obs_id)


@pytest.mark.requires_brms
@pytest.mark.slow
def test_univariate_posterior_epred_insample(
    fit_uni: FitResult, epilepsy_df: pd.DataFrame
) -> None:
    from brmspy import brms

    res = brms.posterior_epred(fit_uni)
    idata: IDPosterior = res.idata
    _assert_groups_exact(idata, {"posterior", "constant_data"})

    obs_id = np.asarray(epilepsy_df.index.to_numpy())

    _assert_dataset_has_vars(idata.posterior, {"count_mean"})
    _assert_da_dims_and_shape(
        idata.posterior["count_mean"],
        dims=("chain", "draw", "obs_id"),
        shape=(C, D, len(epilepsy_df)),
    )
    _assert_obs_id_coords_exact(idata.posterior["count_mean"], obs_id)


@pytest.mark.requires_brms
@pytest.mark.slow
def test_univariate_posterior_epred_newdata_and_obs_id_passthrough(
    fit_uni: FitResult, epilepsy_df: pd.DataFrame, newdata_df: pd.DataFrame
) -> None:
    from brmspy import brms

    res = brms.posterior_epred(fit_uni, newdata=newdata_df)
    idata: IDPredictions = res.idata
    _assert_groups_exact(idata, {"predictions", "predictions_constant_data"})

    pred_obs_id = np.asarray(newdata_df.index.to_numpy())

    _assert_dataset_has_vars(idata.predictions, {"count_mean"})
    _assert_da_dims_and_shape(
        idata.predictions["count_mean"],
        dims=("chain", "draw", "obs_id"),
        shape=(C, D, len(newdata_df)),
    )
    _assert_obs_id_coords_exact(idata.predictions["count_mean"], pred_obs_id)

    _assert_constant_dataset(
        idata.predictions_constant_data,
        expected_obs_id=pred_obs_id,
        must_include={"zAge", "zBase", "Trt", "patient"},
        must_exclude={"count"},
        expected_n=len(newdata_df),
    )


@pytest.mark.requires_brms
@pytest.mark.slow
def test_univariate_posterior_predict_insample(
    fit_uni: FitResult, epilepsy_df: pd.DataFrame
) -> None:
    from brmspy import brms

    res = brms.posterior_predict(fit_uni)
    idata: IDPosteriorPredictive = res.idata
    _assert_groups_exact(idata, {"posterior_predictive", "constant_data"})

    obs_id = np.asarray(epilepsy_df.index.to_numpy())

    _assert_dataset_has_vars(idata.posterior_predictive, {"count"})
    _assert_da_dims_and_shape(
        idata.posterior_predictive["count"],
        dims=("chain", "draw", "obs_id"),
        shape=(C, D, len(epilepsy_df)),
    )
    _assert_obs_id_coords_exact(idata.posterior_predictive["count"], obs_id)


@pytest.mark.requires_brms
@pytest.mark.slow
def test_univariate_posterior_predict_newdata(
    fit_uni: FitResult, newdata_df: pd.DataFrame
) -> None:
    from brmspy import brms

    res = brms.posterior_predict(fit_uni, newdata=newdata_df)
    idata: IDPredictions = res.idata
    _assert_groups_exact(idata, {"predictions", "predictions_constant_data"})

    pred_obs_id = np.asarray(newdata_df.index.to_numpy())

    _assert_dataset_has_vars(idata.predictions, {"count"})
    _assert_da_dims_and_shape(
        idata.predictions["count"],
        dims=("chain", "draw", "obs_id"),
        shape=(C, D, len(newdata_df)),
    )
    _assert_obs_id_coords_exact(idata.predictions["count"], pred_obs_id)


@pytest.mark.requires_brms
@pytest.mark.slow
def test_univariate_posterior_linpred_insample(
    fit_uni: FitResult, epilepsy_df: pd.DataFrame
) -> None:
    from brmspy import brms

    res = brms.posterior_linpred(fit_uni)
    idata: IDPosterior = res.idata
    _assert_groups_exact(idata, {"posterior", "constant_data"})

    obs_id = np.asarray(epilepsy_df.index.to_numpy())

    _assert_dataset_has_vars(idata.posterior, {"count_linpred"})
    _assert_da_dims_and_shape(
        idata.posterior["count_linpred"],
        dims=("chain", "draw", "obs_id"),
        shape=(C, D, len(epilepsy_df)),
    )
    _assert_obs_id_coords_exact(idata.posterior["count_linpred"], obs_id)


@pytest.mark.requires_brms
@pytest.mark.slow
def test_univariate_posterior_linpred_newdata(
    fit_uni: FitResult, newdata_df: pd.DataFrame
) -> None:
    from brmspy import brms

    res = brms.posterior_linpred(fit_uni, newdata=newdata_df)
    idata: IDPredictions = res.idata
    _assert_groups_exact(idata, {"predictions", "predictions_constant_data"})

    pred_obs_id = np.asarray(newdata_df.index.to_numpy())

    _assert_dataset_has_vars(idata.predictions, {"count_linpred"})
    _assert_da_dims_and_shape(
        idata.predictions["count_linpred"],
        dims=("chain", "draw", "obs_id"),
        shape=(C, D, len(newdata_df)),
    )
    _assert_obs_id_coords_exact(idata.predictions["count_linpred"], pred_obs_id)


@pytest.mark.requires_brms
@pytest.mark.slow
def test_univariate_log_lik_insample(
    fit_uni: FitResult, epilepsy_df: pd.DataFrame
) -> None:
    from brmspy import brms

    res = brms.log_lik(fit_uni)
    idata: IDLogLikelihoodInsample = res.idata
    _assert_groups_exact(idata, {"log_likelihood", "constant_data"})

    obs_id = np.asarray(epilepsy_df.index.to_numpy())

    _assert_dataset_has_vars(idata.log_likelihood, {"count"})
    _assert_da_dims_and_shape(
        idata.log_likelihood["count"],
        dims=("chain", "draw", "obs_id"),
        shape=(C, D, len(epilepsy_df)),
    )
    _assert_obs_id_coords_exact(idata.log_likelihood["count"], obs_id)


@pytest.mark.requires_brms
@pytest.mark.slow
def test_univariate_log_lik_newdata_requires_response_column(
    fit_uni: FitResult, newdata_df: pd.DataFrame
) -> None:
    """
    Spec: log_lik(m, newdata) requires response in newdata (univariate: 'count').

    This test is intentionally strict: it should raise if response is absent.
    """
    from brmspy import brms

    new_missing = newdata_df.drop(columns=["count"])
    with pytest.raises(Exception):
        brms.log_lik(fit_uni, newdata=new_missing)


# -----------------------------------------
# Multivariate prediction function contracts
# -----------------------------------------


@pytest.mark.requires_brms
@pytest.mark.slow
def test_multivariate_observed_data_has_both_responses(
    fit_multi: FitResult, epilepsy_df: pd.DataFrame
) -> None:
    from brmspy import brms

    res = brms.observed_data(fit_multi)
    idata: IDObservedData = res.idata
    _assert_groups_exact(idata, {"observed_data", "constant_data"})

    obs_id = np.asarray(epilepsy_df.index.to_numpy())

    _assert_dataset_has_vars(idata.observed_data, {"count", "Base"})
    for resp in ("count", "Base"):
        _assert_da_dims_and_shape(
            idata.observed_data[resp], dims=("obs_id",), shape=(len(epilepsy_df),)
        )
        _assert_obs_id_coords_exact(idata.observed_data[resp], obs_id)


@pytest.mark.requires_brms
@pytest.mark.slow
def test_multivariate_posterior_epred_insample_has_both_means(
    fit_multi: FitResult, epilepsy_df: pd.DataFrame
) -> None:
    from brmspy import brms

    res = brms.posterior_epred(fit_multi)
    idata: IDPosterior = res.idata
    _assert_groups_exact(idata, {"posterior", "constant_data"})

    obs_id = np.asarray(epilepsy_df.index.to_numpy())

    _assert_dataset_has_vars(idata.posterior, {"count_mean", "Base_mean"})
    for v in ("count_mean", "Base_mean"):
        _assert_da_dims_and_shape(
            idata.posterior[v],
            dims=("chain", "draw", "obs_id"),
            shape=(C, D, len(epilepsy_df)),
        )
        _assert_obs_id_coords_exact(idata.posterior[v], obs_id)


@pytest.mark.requires_brms
@pytest.mark.slow
def test_multivariate_posterior_predict_newdata_has_both_responses(
    fit_multi: FitResult, newdata_df: pd.DataFrame
) -> None:
    from brmspy import brms

    res = brms.posterior_predict(fit_multi, newdata=newdata_df)
    idata: IDPredictions = res.idata
    _assert_groups_exact(idata, {"predictions", "predictions_constant_data"})

    pred_obs_id = np.asarray(newdata_df.index.to_numpy())

    _assert_dataset_has_vars(idata.predictions, {"count", "Base"})
    for resp in ("count", "Base"):
        _assert_da_dims_and_shape(
            idata.predictions[resp],
            dims=("chain", "draw", "obs_id"),
            shape=(C, D, len(newdata_df)),
        )
        _assert_obs_id_coords_exact(idata.predictions[resp], pred_obs_id)


@pytest.mark.requires_brms
@pytest.mark.slow
def test_multivariate_posterior_linpred_newdata_has_both_linpreds(
    fit_multi: FitResult, newdata_df: pd.DataFrame
) -> None:
    from brmspy import brms

    res = brms.posterior_linpred(fit_multi, newdata=newdata_df)
    idata: IDPredictions = res.idata
    _assert_groups_exact(idata, {"predictions", "predictions_constant_data"})

    pred_obs_id = np.asarray(newdata_df.index.to_numpy())

    _assert_dataset_has_vars(idata.predictions, {"count_linpred", "Base_linpred"})
    for v in ("count_linpred", "Base_linpred"):
        _assert_da_dims_and_shape(
            idata.predictions[v],
            dims=("chain", "draw", "obs_id"),
            shape=(C, D, len(newdata_df)),
        )
        _assert_obs_id_coords_exact(idata.predictions[v], pred_obs_id)


@pytest.mark.requires_brms
@pytest.mark.slow
def test_multivariate_log_lik_newdata_requires_both_response_columns(
    fit_multi: FitResult, newdata_df: pd.DataFrame
) -> None:
    """
    Spec: log_lik(m, newdata) requires responses in newdata (multivariate: 'count' and 'Base').

    This test is intentionally strict: it should raise if any required response is absent.
    """
    from brmspy import brms

    # missing one response
    missing_base = newdata_df.drop(columns=["Base"])
    with pytest.raises(Exception):
        brms.log_lik(fit_multi, newdata=missing_base)

    missing_count = newdata_df.drop(columns=["count"])
    with pytest.raises(Exception):
        brms.log_lik(fit_multi, newdata=missing_count)


# ----------------------------------------------------
# Composition / return_idata=False style forwardability
# ----------------------------------------------------


@pytest.mark.requires_brms
@pytest.mark.slow
def test_raw_r_fit_handle_is_accepted_by_prediction_functions(
    fit_uni: FitResult,
) -> None:
    """
    Spec: brm(return_idata=False) returns an R brmsfit proxy usable by prediction functions.

    Instead of re-fitting, we use fit_uni.r (the underlying brmsfit handle),
    which should be equivalent to what return_idata=False would return.
    """
    from brmspy import brms

    raw = fit_uni.r
    assert not hasattr(raw, "idata")

    idata_post = brms.posterior(raw).idata
    _assert_groups_exact(idata_post, {"posterior", "constant_data"})

    idata_obs = brms.observed_data(raw).idata
    _assert_groups_exact(idata_obs, {"observed_data", "constant_data"})

    idata_ll = brms.log_lik(raw).idata
    _assert_groups_exact(idata_ll, {"log_likelihood", "constant_data"})
