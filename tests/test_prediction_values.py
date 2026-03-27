"""
Prediction **data validity** tests for brmspy.

Unlike `test_arviz_shapes.py` (which only checks dims/shapes/coord names),
these tests verify that the *values* returned by prediction wrappers are
actually meaningful — i.e. the predictions vary across observations and
reflect the structure of the fitted model.

Design:
  • Use a simple Gaussian model: y ~ group  (two groups with very different
    means: group_A ≈ 0, group_B ≈ 100).
  • After fitting, the posterior mean predictions for group_A rows should be
    clearly different from group_B rows.
  • If the wrapper silently flattens / duplicates / transposes data, all
    predictions will look the same and the test fails.

Requires brms (uses @pytest.mark.requires_brms).
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from brmspy.types.brms_results import FitResult


# ---------------------------------------------------------------------------
# Test data — two groups with maximally distinct means
# ---------------------------------------------------------------------------

N_PER_GROUP = 30
SEED = 42
C = 2
ITER = 400
WARMUP = 200

# The means are deliberately far apart so even short MCMC runs
# produce clearly separated posterior predictions.
GROUP_A_MEAN = 0.0
GROUP_B_MEAN = 100.0
NOISE_SD = 2.0


@pytest.fixture(scope="module")
def distinct_df() -> pd.DataFrame:
    rng = np.random.default_rng(SEED)
    n = N_PER_GROUP
    df = pd.DataFrame(
        {
            "y": np.concatenate(
                [
                    rng.normal(GROUP_A_MEAN, NOISE_SD, n),
                    rng.normal(GROUP_B_MEAN, NOISE_SD, n),
                ]
            ),
            "group": ["A"] * n + ["B"] * n,
        }
    )
    df.index = [f"obs{i}" for i in range(len(df))]
    return df


@pytest.fixture(scope="module")
def fit_distinct(distinct_df: pd.DataFrame) -> FitResult:
    from brmspy import brms

    return brms.brm(
        "y ~ group",
        data=distinct_df,
        chains=C,
        iter=ITER,
        warmup=WARMUP,
        seed=SEED,
        cores=2,
        refresh=0,
        silent=2,
    )


# ---------------------------------------------------------------------------
# Helper — split predictions by group membership
# ---------------------------------------------------------------------------


def _group_means(
    predictions: np.ndarray, groups: np.ndarray
) -> dict[str, float]:
    """
    Given predictions with shape (chain, draw, obs) and a groups array of
    length obs, return the grand mean prediction for each group.
    """
    # Average over chains and draws → per-observation mean
    obs_means = predictions.mean(axis=(0, 1))  # shape (obs,)
    result = {}
    for g in np.unique(groups):
        mask = groups == g
        result[g] = float(obs_means[mask].mean())
    return result


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


@pytest.mark.requires_brms
@pytest.mark.slow
class TestPosteriorEpredDataValidity:
    """posterior_epred should return predictions that reflect group structure."""

    def test_insample_groups_differ(
        self, fit_distinct: FitResult, distinct_df: pd.DataFrame
    ) -> None:
        from brmspy import brms

        res = brms.posterior_epred(fit_distinct)
        idata = res.idata

        # Get the prediction variable (y_mean for univariate)
        ds = idata.posterior
        var_names = list(ds.data_vars)
        assert len(var_names) >= 1, f"No prediction variables: {var_names}"

        pred_var = [v for v in var_names if "mean" in v or v == "y"][0]
        predictions = ds[pred_var].values  # (chain, draw, obs)

        groups = distinct_df["group"].values
        means = _group_means(predictions, groups)

        # The gap between group means should be substantial (true gap is 100)
        gap = abs(means["B"] - means["A"])
        assert gap > 50, (
            f"Expected large gap between group means, got {gap:.1f}. "
            f"Group A mean: {means['A']:.1f}, Group B mean: {means['B']:.1f}. "
            f"Predictions may be flattened/broken."
        )

    def test_newdata_groups_differ(
        self, fit_distinct: FitResult, distinct_df: pd.DataFrame
    ) -> None:
        from brmspy import brms

        # Predict on a small newdata with both groups
        newdata = pd.DataFrame(
            {"group": ["A", "A", "A", "B", "B", "B"]},
            index=[f"new{i}" for i in range(6)],
        )

        res = brms.posterior_epred(fit_distinct, newdata=newdata)
        idata = res.idata

        ds = idata.predictions
        var_names = list(ds.data_vars)
        pred_var = [v for v in var_names if "mean" in v or v == "y"][0]
        predictions = ds[pred_var].values  # (chain, draw, obs)

        groups = newdata["group"].values
        means = _group_means(predictions, groups)

        gap = abs(means["B"] - means["A"])
        assert gap > 50, (
            f"Expected large gap in newdata predictions, got {gap:.1f}. "
            f"Group A: {means['A']:.1f}, Group B: {means['B']:.1f}."
        )


@pytest.mark.requires_brms
@pytest.mark.slow
class TestPosteriorPredictDataValidity:
    """posterior_predict should return predictions that reflect group structure."""

    def test_insample_groups_differ(
        self, fit_distinct: FitResult, distinct_df: pd.DataFrame
    ) -> None:
        from brmspy import brms

        res = brms.posterior_predict(fit_distinct)
        idata = res.idata

        ds = idata.posterior_predictive
        var_names = list(ds.data_vars)
        pred_var = [v for v in var_names if v == "y" or "y" in v][0]
        predictions = ds[pred_var].values

        groups = distinct_df["group"].values
        means = _group_means(predictions, groups)

        gap = abs(means["B"] - means["A"])
        assert gap > 50, (
            f"posterior_predict: gap {gap:.1f} too small. "
            f"A={means['A']:.1f}, B={means['B']:.1f}."
        )

    def test_newdata_groups_differ(
        self, fit_distinct: FitResult
    ) -> None:
        from brmspy import brms

        newdata = pd.DataFrame(
            {"group": ["A", "A", "B", "B"]},
            index=[f"p{i}" for i in range(4)],
        )

        res = brms.posterior_predict(fit_distinct, newdata=newdata)
        idata = res.idata

        ds = idata.predictions
        var_names = list(ds.data_vars)
        pred_var = [v for v in var_names if v == "y" or "y" in v][0]
        predictions = ds[pred_var].values

        groups = newdata["group"].values
        means = _group_means(predictions, groups)

        gap = abs(means["B"] - means["A"])
        assert gap > 50, (
            f"posterior_predict newdata: gap {gap:.1f} too small. "
            f"A={means['A']:.1f}, B={means['B']:.1f}."
        )


@pytest.mark.requires_brms
@pytest.mark.slow
class TestPosteriorLinpredDataValidity:
    """posterior_linpred should return differentiated linear predictor values."""

    def test_insample_groups_differ(
        self, fit_distinct: FitResult, distinct_df: pd.DataFrame
    ) -> None:
        from brmspy import brms

        res = brms.posterior_linpred(fit_distinct)
        idata = res.idata

        ds = idata.posterior
        var_names = list(ds.data_vars)
        pred_var = [v for v in var_names if "linpred" in v or v == "y"][0]
        predictions = ds[pred_var].values

        groups = distinct_df["group"].values
        means = _group_means(predictions, groups)

        gap = abs(means["B"] - means["A"])
        assert gap > 50, (
            f"posterior_linpred: gap {gap:.1f} too small. "
            f"A={means['A']:.1f}, B={means['B']:.1f}."
        )

    def test_newdata_groups_differ(
        self, fit_distinct: FitResult
    ) -> None:
        from brmspy import brms

        newdata = pd.DataFrame(
            {"group": ["A", "A", "B", "B"]},
            index=[f"lp{i}" for i in range(4)],
        )

        res = brms.posterior_linpred(fit_distinct, newdata=newdata)
        idata = res.idata

        ds = idata.predictions
        var_names = list(ds.data_vars)
        pred_var = [v for v in var_names if "linpred" in v or v == "y"][0]
        predictions = ds[pred_var].values

        groups = newdata["group"].values
        means = _group_means(predictions, groups)

        gap = abs(means["B"] - means["A"])
        assert gap > 50, (
            f"posterior_linpred newdata: gap {gap:.1f} too small. "
            f"A={means['A']:.1f}, B={means['B']:.1f}."
        )


@pytest.mark.requires_brms
@pytest.mark.slow
class TestLogLikDataValidity:
    """log_lik values should differ across observations with different y."""

    def test_insample_varies_across_groups(
        self, fit_distinct: FitResult, distinct_df: pd.DataFrame
    ) -> None:
        from brmspy import brms

        res = brms.log_lik(fit_distinct)
        idata = res.idata

        ds = idata.log_likelihood
        var_names = list(ds.data_vars)
        ll_var = var_names[0]
        ll_vals = ds[ll_var].values  # (chain, draw, obs)

        # Mean log-lik per observation (averaged across chains/draws)
        obs_mean_ll = ll_vals.mean(axis=(0, 1))

        # log-lik values should have meaningful variation across obs
        obs_std = obs_mean_ll.std()
        assert obs_std > 0.01, (
            f"log_lik has near-zero variation across observations "
            f"(std={obs_std:.6f}), suggesting flattened data."
        )

    def test_newdata_varies_across_groups(
        self, fit_distinct: FitResult
    ) -> None:
        from brmspy import brms

        # log_lik needs the response column in newdata
        newdata = pd.DataFrame(
            {
                "y": [GROUP_A_MEAN, GROUP_A_MEAN, GROUP_B_MEAN, GROUP_B_MEAN],
                "group": ["A", "A", "B", "B"],
            },
            index=[f"ll{i}" for i in range(4)],
        )

        res = brms.log_lik(fit_distinct, newdata=newdata)
        idata = res.idata

        ds = idata.log_likelihood
        var_names = list(ds.data_vars)
        ll_var = var_names[0]
        ll_vals = ds[ll_var].values

        obs_mean_ll = ll_vals.mean(axis=(0, 1))
        # With y at exact group means, model fits both groups similarly well,
        # so log-lik variation is genuinely small. Just check it's non-zero.
        obs_std = obs_mean_ll.std()
        assert obs_std > 0.001, (
            f"log_lik newdata: near-zero variation "
            f"(std={obs_std:.6f}), suggesting flattened data."
        )
