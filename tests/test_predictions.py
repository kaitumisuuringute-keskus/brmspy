"""
Non-shape tests for brmspy prediction functions.

The comprehensive ArviZ shape/spec contract is enforced in
[`tests/test_arviz_shapes.py`](tests/test_arviz_shapes.py).

This file intentionally keeps only tests that are NOT covered by the shape suite:
- smoke coverage for the string-formula + gaussian-family path (brms.fit alias)
- behavioral check: posterior_predict includes observation noise vs posterior_epred
- Poisson posterior_predict draws are integer-ish
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest


@pytest.mark.requires_brms
class TestPredictionBehavior:
    """Behavioral tests not covered by the ArviZ shape suite."""

    @pytest.mark.slow
    def test_string_formula_gaussian_smoke(
        self, sample_dataframe: pd.DataFrame
    ) -> None:
        """
        Ensure the string-formula + gaussian-family path works end-to-end.

        This covers `brms.fit(...)` / `brms.brm(str, family="gaussian")`, which is not
        exercised by the ArviZ shape suite (it uses the formula DSL and Poisson).
        """
        from brmspy import brms
        import arviz as az

        model = brms.fit(
            formula="y ~ x1",
            data=sample_dataframe,
            family="gaussian",
            iter=200,
            warmup=100,
            chains=2,
            seed=123,
            silent=2,
            refresh=0,
        )

        assert isinstance(model.idata, az.InferenceData)

        # Basic smoke calls (structure is asserted in the ArviZ shape suite)
        assert brms.posterior_epred(model, newdata=sample_dataframe).idata is not None
        assert brms.posterior_predict(model, newdata=sample_dataframe).idata is not None
        assert brms.posterior_linpred(model, newdata=sample_dataframe).idata is not None
        assert brms.log_lik(model, newdata=sample_dataframe).idata is not None

    @pytest.mark.slow
    def test_epred_vs_predict_difference(self, sample_dataframe: pd.DataFrame) -> None:
        """
        Epred should give an expected value (mean) while predict adds noise.

        Therefore posterior_predict should have higher variance than posterior_epred.
        """
        from brmspy import brms

        model = brms.fit(
            formula="y ~ x1",
            data=sample_dataframe,
            family="gaussian",
            iter=300,
            warmup=150,
            chains=2,
            seed=123,
            silent=2,
            refresh=0,
        )

        epred = brms.posterior_epred(model=model, newdata=sample_dataframe)
        predict = brms.posterior_predict(model=model, newdata=sample_dataframe)

        epred_vals = epred.idata.predictions["y_mean"].values
        predict_vals = predict.idata.predictions["y"].values

        epred_std = float(np.std(epred_vals))
        predict_std = float(np.std(predict_vals))

        assert predict_std > epred_std, (
            f"Predictions (std={predict_std:.3f}) should have higher variance "
            f"than expected values (std={epred_std:.3f})"
        )


@pytest.mark.requires_brms
class TestPoissonPredictions:
    """Family-specific behavior tests not covered by the ArviZ shape suite."""

    @pytest.mark.slow
    def test_poisson_predictions_are_integerish(
        self, poisson_data: pd.DataFrame
    ) -> None:
        """For Poisson: posterior_predict draws should be integer-ish."""
        from brmspy import brms

        model = brms.fit(
            formula="count ~ predictor",
            data=poisson_data,
            family="poisson",
            iter=200,
            warmup=100,
            chains=2,
            seed=123,
            silent=2,
            refresh=0,
        )

        predict = brms.posterior_predict(model=model, newdata=poisson_data)

        predict_vals = predict.idata.predictions["count"].values

        # At least some counts should be positive
        assert np.any(predict_vals > 0)

        # Integer-ish (allowing float representation)
        assert np.allclose(predict_vals, np.round(predict_vals), atol=1e-10)
