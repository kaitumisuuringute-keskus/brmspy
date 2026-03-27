"""
Tests for the ArviZ version compatibility layer.

These tests verify that ``brmspy.helpers.arviz_compat`` correctly abstracts
over ArviZ < 1.0 (InferenceData) and >= 1.0 (DataTree).  They run against
whichever ArviZ version is installed — so CI should execute them twice:
once with ArviZ < 1.0 and once with ArviZ >= 1.0.

No R or brms is required.
"""

from __future__ import annotations

import numpy as np
import pytest
import xarray as xr

from brmspy.helpers.arviz_compat import (
    ARVIZ_V1,
    InferenceDataBase,
    InferenceDataLike,
    construct_from_datasets,
    extend,
    from_dict,
    get_group_dataset,
    get_groups,
    has_group,
    is_inference_data,
    set_group_dataset,
)


# ---------------------------------------------------------------------------
# Version detection
# ---------------------------------------------------------------------------


class TestVersionDetection:
    def test_arviz_v1_is_bool(self):
        assert isinstance(ARVIZ_V1, bool)

    def test_version_matches_installed(self):
        import arviz as az
        from packaging.version import Version

        expected = Version(az.__version__) >= Version("1.0.0")
        assert ARVIZ_V1 == expected

    def test_inference_data_base_is_correct_type(self):
        if ARVIZ_V1:
            assert InferenceDataBase is xr.DataTree
        else:
            import arviz as az

            assert InferenceDataBase is az.InferenceData

    def test_inference_data_like_matches_base(self):
        # InferenceDataLike should be the same concrete type as the base
        assert InferenceDataLike is InferenceDataBase


# ---------------------------------------------------------------------------
# from_dict
# ---------------------------------------------------------------------------


class TestFromDict:
    def test_returns_expected_type(self):
        obj = from_dict(posterior={"x": np.ones((2, 3, 5))})
        assert isinstance(obj, InferenceDataBase)

    def test_single_group(self):
        obj = from_dict(posterior={"x": np.ones((2, 3, 5))})
        groups = get_groups(obj)
        assert "posterior" in groups

    def test_multiple_groups(self):
        obj = from_dict(
            posterior={"x": np.ones((2, 3, 5))},
            observed_data={"y": np.array([1, 2, 3])},
        )
        groups = set(get_groups(obj))
        assert {"posterior", "observed_data"}.issubset(groups)


# ---------------------------------------------------------------------------
# get_groups
# ---------------------------------------------------------------------------


class TestGetGroups:
    def test_returns_list_of_strings(self):
        obj = from_dict(posterior={"x": np.ones((2, 3, 5))})
        groups = get_groups(obj)
        assert isinstance(groups, list)
        assert all(isinstance(g, str) for g in groups)

    def test_no_slash_prefix(self):
        """Group names should be plain (no leading /)."""
        obj = from_dict(
            posterior={"x": np.ones((2, 3, 5))},
            observed_data={"y": np.array([1.0, 2.0])},
        )
        for name in get_groups(obj):
            assert not name.startswith("/"), f"Group name starts with /: {name!r}"


# ---------------------------------------------------------------------------
# extend
# ---------------------------------------------------------------------------


class TestExtend:
    def test_extend_adds_group(self):
        obj = from_dict(posterior={"x": np.ones((2, 3, 5))})
        other = from_dict(observed_data={"y": np.array([1, 2, 3])})
        extend(obj, other)

        groups = set(get_groups(obj))
        assert "observed_data" in groups
        assert "posterior" in groups

    def test_extend_preserves_existing_data(self):
        obj = from_dict(posterior={"x": np.ones((2, 3, 5))})
        other = from_dict(observed_data={"y": np.array([1, 2, 3])})
        extend(obj, other)

        ds = get_group_dataset(obj, "posterior")
        assert "x" in ds.data_vars


# ---------------------------------------------------------------------------
# get / set group dataset
# ---------------------------------------------------------------------------


class TestGroupDataset:
    def test_get_group_returns_dataset(self):
        obj = from_dict(posterior={"x": np.ones((2, 3, 5))})
        ds = get_group_dataset(obj, "posterior")
        # DatasetView (on DataTree) is a Dataset subclass
        assert isinstance(ds, xr.Dataset)

    def test_get_group_has_expected_vars(self):
        obj = from_dict(posterior={"x": np.ones((2, 3, 5))})
        ds = get_group_dataset(obj, "posterior")
        assert "x" in ds.data_vars

    def test_set_group_replaces_dataset(self):
        obj = from_dict(posterior={"x": np.ones((2, 3, 5))})
        new_ds = xr.Dataset({"z": (["chain", "draw"], np.zeros((2, 3)))})
        set_group_dataset(obj, "posterior", new_ds)

        ds = get_group_dataset(obj, "posterior")
        assert "z" in ds.data_vars


# ---------------------------------------------------------------------------
# has_group
# ---------------------------------------------------------------------------


class TestHasGroup:
    def test_existing_group(self):
        obj = from_dict(posterior={"x": np.ones((2, 3, 5))})
        assert has_group(obj, "posterior") is True

    def test_missing_group(self):
        obj = from_dict(posterior={"x": np.ones((2, 3, 5))})
        assert has_group(obj, "observed_data") is False


# ---------------------------------------------------------------------------
# construct_from_datasets
# ---------------------------------------------------------------------------


class TestConstructFromDatasets:
    def test_round_trip(self):
        ds_post = xr.Dataset(
            {"x": (["chain", "draw"], np.ones((2, 3)))}
        )
        ds_obs = xr.Dataset({"y": (["obs_id"], np.array([1, 2, 3]))})

        obj = construct_from_datasets(
            {"posterior": ds_post, "observed_data": ds_obs}
        )

        assert isinstance(obj, InferenceDataBase)
        groups = set(get_groups(obj))
        assert {"posterior", "observed_data"}.issubset(groups)

    def test_reconstructed_data_matches(self):
        vals = np.array([10.0, 20.0, 30.0])
        ds = xr.Dataset({"y": (["obs_id"], vals)})
        obj = construct_from_datasets({"observed_data": ds})

        ds_out = get_group_dataset(obj, "observed_data")
        np.testing.assert_array_equal(ds_out["y"].values, vals)


# ---------------------------------------------------------------------------
# is_inference_data
# ---------------------------------------------------------------------------


class TestIsInferenceData:
    def test_positive(self):
        obj = from_dict(posterior={"x": np.ones((2, 3, 5))})
        assert is_inference_data(obj) is True

    def test_negative_int(self):
        assert is_inference_data(42) is False

    def test_negative_dataset(self):
        ds = xr.Dataset({"x": (["a"], [1, 2, 3])})
        assert is_inference_data(ds) is False


# ---------------------------------------------------------------------------
# Type hierarchy integration
# ---------------------------------------------------------------------------


class TestTypeHierarchy:
    def test_id_classes_subclass_base(self):
        from brmspy.types.brms_results import (
            IDBrm,
            IDConstantData,
            IDPosterior,
            IDPredictions,
        )

        for cls in (IDConstantData, IDPosterior, IDPredictions, IDBrm):
            assert issubclass(cls, InferenceDataBase), (
                f"{cls.__name__} should subclass {InferenceDataBase.__name__}"
            )

    def test_cast_to_id_brm(self):
        """The cast(IDBrm, idata) pattern used in _conversion.py should work."""
        from typing import cast

        from brmspy.types.brms_results import IDBrm

        obj = from_dict(posterior={"x": np.ones((2, 3, 5))})
        result = cast(IDBrm, obj)
        # cast is a no-op at runtime, so result should be the same object
        assert result is obj
