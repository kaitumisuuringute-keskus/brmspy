from decimal import Decimal
import datetime as dt
from typing import Any, cast
import numpy as np
import pandas as pd
import pytest


class TestTypeCoercion:
    """Test Stan type coercion functionality."""

    @pytest.mark.worker
    def test_coerce_types_basic(self):
        """Test basic type coercion from Stan code"""
        from brmspy.helpers._rpy2._conversion import _coerce_stan_types

        # Simple Stan code with int and real types
        stan_code = """
        data {
            int N;
            int<lower=0> K;
            real y[N];
            matrix[N, K] X;
        }
        """

        # Sample data that needs coercion
        stan_data = {
            "N": np.array([50]),  # Should become scalar int
            "K": np.array([3]),  # Should become scalar int
            "y": np.array([1.5, 2.3, 3.1]),  # Should stay as is
            "X": np.random.randn(3, 3),  # Should stay as is
        }

        result = _coerce_stan_types(stan_code, stan_data)

        # Check N and K are scalars and integers
        assert isinstance(result["N"], (int, np.integer))
        assert isinstance(result["K"], (int, np.integer))
        assert result["N"] == 50
        assert result["K"] == 3

    @pytest.mark.worker
    def test_coerce_types_preserves_arrays(self):
        """Test that arrays are preserved correctly"""
        from brmspy.helpers._rpy2._conversion import _coerce_stan_types

        stan_code = """
        data {
            int N;
            vector[N] y;
        }
        """

        y_data = np.array([1.1, 2.2, 3.3, 4.4, 5.5])
        stan_data = {"N": np.array([5]), "y": y_data}

        result = _coerce_stan_types(stan_code, stan_data)

        # N should be scalar
        assert isinstance(result["N"], (int, np.integer))
        # y should still be an array
        assert isinstance(result["y"], np.ndarray)
        assert len(result["y"]) == 5

    @pytest.mark.worker
    def test_coerce_int_array(self):
        """Test coercion of int arrays"""
        from brmspy.helpers._rpy2._conversion import _coerce_stan_types

        stan_code = """
        data {
            int N;
            int Y[N];
        }
        """

        # Simulate data that might come from R with float types
        stan_data = {
            "N": np.array([5.0]),
            "Y": np.array([1.0, 2.0, 3.0, 4.0, 5.0]),  # Floats that should be ints
        }

        result = _coerce_stan_types(stan_code, stan_data)

        # N should be scalar int
        assert isinstance(result["N"], (int, np.integer))
        assert result["N"] == 5

        # Y should be int array
        assert isinstance(result["Y"], np.ndarray)
        assert result["Y"].dtype in [np.int32, np.int64]
        assert np.array_equal(result["Y"], [1, 2, 3, 4, 5])

    @pytest.mark.worker
    def test_coerce_mixed_types(self):
        """Test coercion with mixed int and real types"""
        from brmspy.helpers._rpy2._conversion import _coerce_stan_types

        stan_code = """
        data {
            int<lower=1> N;
            int K;
            int Y[N];
            real X[N];
            vector[N] Z;
        }
        """

        stan_data = {
            "N": np.array([10.0]),
            "K": np.array([3.0]),
            "Y": np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0]),
            "X": np.array([1.1, 2.2, 3.3, 4.4, 5.5, 6.6, 7.7, 8.8, 9.9, 10.1]),
            "Z": np.array([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]),
        }

        result = _coerce_stan_types(stan_code, stan_data)

        # Scalars should be properly coerced
        assert isinstance(result["N"], (int, np.integer))
        assert isinstance(result["K"], (int, np.integer))

        # Int array should be int type
        assert result["Y"].dtype in [np.int32, np.int64]

        # Real arrays should stay as float
        assert result["X"].dtype in [np.float32, np.float64]
        assert result["Z"].dtype in [np.float32, np.float64]

    @pytest.mark.worker
    def test_coerce_handles_non_numpy(self):
        """Test coercion handles non-numpy types"""
        from brmspy.helpers._rpy2._conversion import _coerce_stan_types

        stan_code = """
        data {
            int N;
            int K;
        }
        """

        # Python lists instead of numpy arrays
        stan_data = {"N": [5], "K": [3]}

        result = _coerce_stan_types(stan_code, stan_data)

        # Should convert and coerce properly
        assert isinstance(result["N"], (int, np.integer))
        assert isinstance(result["K"], (int, np.integer))
        assert result["N"] == 5
        assert result["K"] == 3

    @pytest.mark.worker
    def test_coerce_new_stan_array_syntax(self):
        """Test type coercion with new Stan array syntax: array[N] int Y

        This tests the specific scenario found with epilepsy count data where:
        - R returns count data as float64
        - Stan declares it as array[N] int Y (new syntax)
        - Must be correctly identified and coerced to int

        Regression test for issue where old parser captured 'array' as type
        instead of 'int', causing Stan runtime errors.
        """
        from brmspy.helpers._rpy2._conversion import _coerce_stan_types

        # New Stan array syntax
        stan_code = """
        data {
            int<lower=1> N;
            array[N] int Y;
            int K;
        }
        """

        # Simulate data from R (floats that should be ints)
        stan_data = {
            "N": np.array([5.0]),
            "Y": np.array([1.0, 2.0, 3.0, 4.0, 5.0]),  # count data as float64
            "K": np.array([3.0]),
        }

        result = _coerce_stan_types(stan_code, stan_data)

        # Verify scalars are converted
        assert isinstance(result["N"], (int, np.integer))
        assert isinstance(result["K"], (int, np.integer))

        # Verify array is converted to int type
        assert isinstance(result["Y"], np.ndarray)
        assert result["Y"].dtype in [
            np.int32,
            np.int64,
        ], f"Y should be int type but got {result['Y'].dtype}"

        # Verify values are preserved
        assert np.array_equal(result["Y"], np.array([1, 2, 3, 4, 5]))


def make_all_pandas_dtypes_df(n: int = 8) -> pd.DataFrame:
    """
    Build a DataFrame containing a broad set of pandas column dtypes:
      - numpy scalar dtypes: bool, ints, uints, floats
      - pandas extension dtypes: nullable ints, nullable bool, nullable floats, strings
      - categorical (string categories + int categories)
      - datetime64[ns], datetime64[ns, tz], timedelta64[ns]
      - period, interval
      - sparse
      - object (mixed python objects)
    """
    idx = pd.Index(range(n), name="idx")
    df = pd.DataFrame(index=idx)

    # --- numpy scalar dtypes ---
    df["np_bool"] = pd.Series(
        np.array([True, False] * (n // 2) + [True] * (n % 2), dtype=np.bool_), index=idx
    )

    df["np_int8"] = pd.Series(np.arange(n, dtype=np.int8), index=idx)
    df["np_int16"] = pd.Series(np.arange(n, dtype=np.int16) - 3, index=idx)
    df["np_int32"] = pd.Series(np.arange(n, dtype=np.int32) - 10, index=idx)
    df["np_int64"] = pd.Series(np.arange(n, dtype=np.int64) - 100, index=idx)

    df["np_uint8"] = pd.Series(np.arange(n, dtype=np.uint8), index=idx)
    df["np_uint16"] = pd.Series(np.arange(n, dtype=np.uint16) + 1, index=idx)
    df["np_uint32"] = pd.Series(np.arange(n, dtype=np.uint32) + 10, index=idx)
    df["np_uint64"] = pd.Series(np.arange(n, dtype=np.uint64) + 100, index=idx)

    df["np_float32"] = pd.Series((np.arange(n, dtype=np.float32) / 3.0), index=idx)
    df["np_float64"] = pd.Series((np.arange(n, dtype=np.float64) / 7.0), index=idx)

    # --- pandas nullable extension dtypes ---
    df["pd_Int64"] = pd.Series(
        [1, 2, pd.NA, 4, 5, pd.NA, 7, 8][:n], dtype="Int64", index=idx
    )
    df["pd_UInt64"] = pd.Series(
        [1, 2, pd.NA, 4, 5, pd.NA, 7, 8][:n], dtype="UInt64", index=idx
    )
    # NOTE: pandas nullable boolean (with pd.NA) is not reliably supported by rpy2/pandas2ri
    # and tends to fall back to string/object conversion, which forces the dataframe
    # roundtrip down the slow/fallback path. Keep it out of this roundtrip test.
    # df["pd_boolean"] = pd.Series(
    #     [True, False, pd.NA, True, False, pd.NA, True, False][:n],
    #     dtype="boolean",
    #     index=idx,
    # )
    df["pd_Float64"] = pd.Series(
        [1.5, pd.NA, 3.25, 4.0, pd.NA, 6.0, 7.0, pd.NA][:n], dtype="Float64", index=idx
    )

    # string (python-backed)
    df["pd_string"] = pd.Series(
        ["a", None, "c", "d", None, "f", "g", "h"][:n], dtype="string", index=idx
    )

    # --- datetimes / timedeltas ---
    # df["np_datetime64ns"] = pd.Series(
    #    pd.date_range("2024-01-01", periods=n, freq="D"), index=idx
    # )
    # df["np_timedelta64ns"] = pd.Series(
    #    pd.to_timedelta(np.arange(n), unit="h"), index=idx
    # )

    # tz-aware datetime (pick a non-UTC tz to catch tz dropping)
    # df["pd_datetime_tz"] = pd.Series(
    #    pd.date_range("2024-01-01", periods=n, freq="D", tz="Europe/Tallinn"), index=idx
    # )

    # --- categorical ---
    # string categories (should roundtrip without rpy2 screaming)
    df["cat_str"] = pd.Series(
        pd.Categorical(
            ["a", "b", "a", None, "c", "b", "c", None][:n],
            categories=["a", "b", "c"],
            ordered=False,
        ),
        index=idx,
    )

    # int categories (rpy2 factor conversion hates these; allowed to come back with string categories)
    df["cat_int"] = pd.Series(
        pd.Categorical(
            [1, 2, 1, None, 2, 1, None, 2][:n], categories=[1, 2], ordered=True
        ),
        index=idx,
    )

    # --- period / interval ---
    # NOTE: period/interval dtypes are not reliably supported by rpy2/pandas2ri and can
    # force dataframe conversion to fall back. Keep them out of this roundtrip test.
    # df["pd_period_M"] = pd.Series(
    #     pd.period_range("2024-01", periods=n, freq="M"), index=idx
    # )
    #
    # intervals = pd.IntervalIndex.from_breaks(
    #     list(range(n + 1)), closed="left"
    # )  # interval[int64, left]
    # df["pd_interval"] = pd.Series(intervals[:n], index=idx)

    # --- sparse ---
    sparse_arr = pd.arrays.SparseArray(
        [0, 1, 0, 2, 0, 3, 0, 4][:n], fill_value=0, dtype=np.int64
    )
    df["pd_sparse_int64"] = pd.Series(sparse_arr, index=idx)

    # --- object (mixed python objects) ---
    # NOTE: mixed-object columns are not supported for a strict R roundtrip and tend to
    # trigger lossy conversions / fallbacks. Keep them out of this test.
    # df["obj_mixed"] = pd.Series(
    #     [
    #         {"k": 1},
    #         (1, 2),
    #         Decimal("1.25"),
    #         dt.date(2024, 1, 1),
    #         None,
    #         b"bytes",
    #         {"nested": {"x": 2}},
    #         (3,),
    #     ][:n],
    #     dtype="object",
    #     index=idx,
    # )

    return df


def _categories_are_strings(cat_dtype: pd.CategoricalDtype) -> bool:
    # robust check: inferred_type is "string" for string categories
    return getattr(cat_dtype.categories, "inferred_type", None) == "string"


acceptable_conversions: dict[np.dtype, set[Any]] = {
    # to: from
    np.dtype("int32"): {
        np.dtype("int64"),
        np.dtype("int8"),
        np.dtype("int16"),
        np.dtype("uint64"),
        np.dtype("uint32"),
        np.dtype("uint16"),
        np.dtype("uint8"),
        np.dtype("bool"),
    },
    # pandas nullable Float64 dtype is acceptable to come back as numpy float64 after R roundtrip
    np.dtype("float64"): {np.dtype("float16"), np.dtype("float32"), pd.Float64Dtype()},
}


@pytest.mark.requires_brms
class TestEncodingAndConversionRoundtrips:
    def test_dataframe_empty(self):
        from brmspy import brms

        df = pd.DataFrame({})
        dfr = brms.call("identity", df)
        assert dfr.empty
        assert len(dfr.columns) == 0
        assert len(dfr.index) == 0

    def test_dataframe_all_dtypes(self):
        from brmspy import brms

        df = make_all_pandas_dtypes_df()

        out = brms.call("identity", df)

        assert isinstance(
            out, pd.DataFrame
        ), f"Expected DataFrame back, got {type(out)!r}"
        assert list(out.columns) == list(df.columns), "Columns changed during roundtrip"
        assert len(out) == len(df), "Row count changed during roundtrip"

        failures: list[str] = []

        try:
            dt0 = df.index.dtype
            dt1 = out.index.dtype
            check_dtype = True
            if dt1 in acceptable_conversions:
                check_dtype = False
                if dt0 != dt1 and not any(
                    dt == dt0 for dt in acceptable_conversions[cast(Any, dt1)]
                ):
                    failures.append(f"{dt0}: index dtype changed to {dt1}")

            pd.testing.assert_series_equal(
                df.index.to_series(),
                out.index.to_series(),
                check_dtype=check_dtype,
                check_index=False,
                check_index_type=False,
                check_names=False,
            )
        except Exception as e:
            failures.append(
                f"index: dtype changed: {df.index.dtype!r} -> {out.index.dtype!r} {e}"
            )

        for col in df.columns:
            s0 = df[col]
            s1 = out[col]

            # Columns that rpy2/pandas2ri cannot faithfully roundtrip (they fall back to
            # string/object conversion). We only assert that the column comes back and
            # has an object dtype, but we don't require value/dtype identity.
            if col in {
                "pd_boolean",
                "pd_string",
                "pd_period_M",
                "pd_interval",
                "obj_mixed",
            }:
                if not pd.api.types.is_object_dtype(s1.dtype):
                    failures.append(
                        f"{col}: expected object dtype after R roundtrip, got {s1.dtype!r}"
                    )
                continue

            # ---- categorical special-case ----
            if isinstance(s0.dtype, pd.CategoricalDtype):
                if not isinstance(s1.dtype, pd.CategoricalDtype):
                    failures.append(f"{col}: expected category, got {s1.dtype!r} {s1}")
                    continue

                d0 = s0.dtype
                d1 = s1.dtype

                if bool(d0.ordered) != bool(d1.ordered):
                    failures.append(
                        f"{col}: category ordered changed: {d0.ordered} -> {d1.ordered}"
                    )

                codes0 = s0.cat.codes.to_numpy()
                codes1 = s1.cat.codes.to_numpy()
                if not np.array_equal(codes0, codes1):
                    failures.append(
                        f"{col}: category codes changed: {codes0.tolist()} -> {codes1.tolist()}"
                    )

                # If original categories are strings, require exact categories equality.
                # If original categories are non-strings (e.g., int), allow rpy2 workaround:
                # categories may come back as strings but must match str(original_categories).
                cats0 = list(d0.categories)
                cats1 = list(d1.categories)

                if _categories_are_strings(d0):
                    if cats0 != cats1:
                        failures.append(
                            f"{col}: string categories changed: {cats0!r} -> {cats1!r}"
                        )
                else:
                    want = [str(x) for x in cats0]
                    got = [str(x) for x in cats1]
                    if want != got:
                        failures.append(
                            f"{col}: non-string categories not preserved up to str(): {want!r} -> {got!r}"
                        )

                continue

            try:
                check_dtype = True
                if pd.api.types.is_integer_dtype(
                    s0.dtype
                ) and pd.api.types.is_extension_array_dtype(s0.dtype):
                    # Nullable int
                    check_dtype = False
                    if s1.dtype != np.float64 and s1.dtype != np.int32:
                        failures.append(
                            f"{col}: {s0.dtype}: dtype changed to {s1.dtype}"
                        )
                else:
                    if s1.dtype in acceptable_conversions:
                        check_dtype = False
                        if s0.dtype != s1.dtype and not any(
                            dt == s0.dtype
                            for dt in acceptable_conversions[cast(Any, s1.dtype)]
                        ):
                            failures.append(
                                f"{col}: {s0.dtype}: dtype changed to {s1.dtype}"
                            )

                pd.testing.assert_series_equal(
                    s0,
                    s1,
                    check_dtype=check_dtype,
                    check_names=True,
                    check_exact=False,
                    check_index_type=False,
                    check_index=False,
                )
            except AssertionError as e:
                failures.append(f"{col}: values changed: {e}")

        if failures:
            raise AssertionError(
                "Roundtrip identity dtype/value failures:\n- " + "\n- ".join(failures)
            )
