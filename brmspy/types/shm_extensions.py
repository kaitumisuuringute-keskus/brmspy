"""
SHM-backed NumPy and pandas helpers.

These types are thin wrappers around NumPy/pandas objects that keep a reference
to the shared-memory block that backs the underlying data. They enable brmspy's
codecs to avoid extra copies when transporting large numeric payloads between
the main process and the worker.

See Also
--------
[`brmspy._session.codec.builtin.NumpyArrayCodec`][brmspy._session.codec.builtin.NumpyArrayCodec]
    Encodes/decodes NumPy arrays into shared memory.
[`brmspy._session.codec.builtin.PandasDFCodec`][brmspy._session.codec.builtin.PandasDFCodec]
    Encodes/decodes DataFrames into shared memory.
[`brmspy.types.shm`][brmspy.types.shm]
    Base shared-memory block and pool types.
"""

import pickle
from typing import Any, Hashable, Literal, Sequence, TypedDict, Union, cast
from numpy.typing import ArrayLike

import numpy as np
import pandas as pd
from zmq import DraftFDWarning

from brmspy.helpers.log import log_warning
from brmspy.types.shm import ShmBlock, ShmRef

__all__ = ["ShmArray", "ShmDataFrameSimple", "ShmDataFrameColumns"]


class ShmArrayMetadata(TypedDict):
    # {"dtype": dtype, "shape": shape, "order": order}
    dtype: str
    shape: list[int]
    order: Literal["C", "F", "non-contiguous"]


class ShmSeriesMetadata(TypedDict):
    name: Hashable | None
    np_dtype: str
    pd_dtype: str
    params: dict[str, Any]  # extra info per logical type
    block: ShmRef


class ShmArray(np.ndarray):
    """
    NumPy array view backed by a shared-memory block.

    Attributes
    ----------
    block : ShmRef
        Reference to the shared-memory block backing the array data.

    Notes
    -----
    This is a *view* over `SharedMemory.buf`. Closing/unlinking the underlying
    shared memory while the array is still in use will lead to undefined
    behavior.
    """

    _shm_metadata: ShmRef  # for type checkers

    @classmethod
    def from_metadata(
        cls, meta: ShmArrayMetadata | dict[str, Any], block: ShmBlock
    ) -> np.ndarray:
        dtype = np.dtype(meta["dtype"])
        shape = tuple(meta["shape"])
        order = meta["order"]

        return ShmArray.from_block(block=block, shape=shape, dtype=dtype, order=order)

    @classmethod
    def from_block(
        cls, block: ShmBlock, shape: tuple[int, ...], dtype: np.dtype, **kwargs
    ) -> Union["ShmArray", np.ndarray]:
        """
        Create an array view backed by an existing shared-memory block.

        Parameters
        ----------
        block : ShmBlock
            Attached shared-memory block.
        shape : tuple[int, ...]
            Desired array shape.
        dtype : numpy.dtype
            NumPy dtype of the array.
        **kwargs
            Reserved for future compatibility. Currently unused.

        Returns
        -------
        ShmArray
            Array view into the shared-memory buffer.
        """
        is_object = dtype == "O"

        if not is_object:
            if block.shm.buf:
                view = memoryview(block.shm.buf)
                view = view[: block.content_size]
            else:
                view = None
            base = np.ndarray(
                shape=shape,
                dtype=dtype,
                buffer=view,
                order=kwargs.get("order", "F"),
            )
            obj = base.view(ShmArray)
            obj._shm_metadata = {
                "name": block.name,
                "size": block.size,
                "content_size": block.content_size,
            }
        else:
            assert block.shm.buf
            view = memoryview(block.shm.buf)
            view = view[: block.content_size]
            payload = bytes(view)
            obj = pickle.loads(payload)
            assert isinstance(obj, np.ndarray)

        return obj

    @classmethod
    def array_order(cls, a: np.ndarray) -> Literal["C", "F", "non-contiguous"]:
        """
        Determine how an array can be reconstructed from a raw buffer.

        Returns `"C"` for C-contiguous arrays, `"F"` for Fortran-contiguous arrays,
        otherwise `"non-contiguous"` (meaning: bytes were obtained by forcing
        a contiguous copy during encoding).
        """
        if a.flags["C_CONTIGUOUS"]:
            return "C"
        if a.flags["F_CONTIGUOUS"]:
            return "F"
        return "non-contiguous"

    @classmethod
    def is_string_object(cls, a: np.ndarray, sample: int = 1000):
        if a.dtype != object:
            return False
        it = a.flat
        for _ in range(min(sample, a.size)):
            v = next(it, None)
            if v is not None and not isinstance(v, str):
                return False
        return True

    @classmethod
    def to_shm(
        cls, obj: np.ndarray | pd.Series | list, shm_pool: Any
    ) -> tuple[np.ndarray | None, ShmRef, str, list[int], str]:
        if isinstance(obj, pd.Series):
            if isinstance(obj.dtype, pd.CategoricalDtype):
                arr = obj.cat.codes.to_numpy(copy=False)
            else:
                arr = obj.to_numpy(copy=False)
        else:
            arr = np.asarray(obj)

        is_object = arr.dtype == "O"
        is_string = cls.is_string_object(arr)

        arr_modified = None

        if isinstance(arr, ShmArray):
            arr = arr
            nbytes = arr._shm_metadata["content_size"]
            ref = arr._shm_metadata

        else:
            if not is_object:
                data = arr.tobytes(order="C")
            elif is_string:
                arr = arr.astype("U")
                arr_modified = arr
                data = arr.tobytes(order="C")
            else:
                data = pickle.dumps(arr, protocol=pickle.HIGHEST_PROTOCOL)

            nbytes = len(data)

            # Ask for exactly nbytes; OS may round up internally, that's fine.
            block = shm_pool.alloc(nbytes)
            block.shm.buf[:nbytes] = data
            ref = ShmRef(
                name=block.name, size=block.size, content_size=block.content_size
            )

        ref, dtype, shape, order = (
            ref,
            str((arr_modified or arr).dtype),
            list(arr.shape),
            cls.array_order(arr),
        )

        return arr_modified, ref, dtype, shape, order


class ShmDataFrameSimple(pd.DataFrame):
    """
    pandas DataFrame backed by a single shared-memory block (numeric only).

    Attributes
    ----------
    block : ShmRef
        Reference to the shared-memory block backing the DataFrame's values.
    """

    _metadata = ["_shm_metadata"]
    _shm_metadata: ShmRef

    @classmethod
    def from_block(
        cls,
        block: ShmBlock,
        nrows: int,
        ncols: int,
        columns: list[Any] | None,
        index: list[Any] | None,
        dtype: str | np.dtype,
    ) -> "ShmDataFrameSimple":
        """
        Construct a DataFrame backed by a single SHM block.

        Parameters
        ----------
        block : ShmBlock
            Attached shared-memory block containing a contiguous 2D numeric matrix.
        nrows, ncols : int
            DataFrame shape.
        columns, index : list[Any] or None
            Column/index labels.
        dtype : str or numpy.dtype
            Dtype of the matrix stored in the block.

        Returns
        -------
        ShmDataFrameSimple
        """
        _dtype = np.dtype(dtype)
        arr = ShmArray.from_block(shape=(ncols, nrows), dtype=_dtype, block=block)

        df = ShmDataFrameSimple(data=arr.T, index=index, columns=columns)
        df._set_shm_metadata(
            {
                "name": block.name,
                "size": block.size,
                "content_size": block.content_size,
            }
        )
        return df

    def _set_shm_metadata(self, meta: ShmRef):
        self._shm_metadata = meta


class ShmDataFrameColumns(pd.DataFrame):
    """
    pandas DataFrame backed by per-column shared-memory blocks (numeric only).

    Attributes
    ----------
    _blocks_columns : dict[str, PandasColumnMetadata]
        Mapping from column name to data required for its reconstruction
    """

    _metadata = ["_shm_metadata"]
    _shm_metadata: dict[str, ShmSeriesMetadata]

    @classmethod
    def _create_col_metadata(
        cls, series: pd.Series, block: ShmRef, arr: np.ndarray | None = None, **params
    ) -> ShmSeriesMetadata:
        pd_dtype = series.dtype

        # Decide what ndarray is actually stored in SHM
        if isinstance(pd_dtype, pd.CategoricalDtype):
            # store dtype params
            params["categories"] = pd_dtype.categories.to_numpy(dtype=object).tolist()
            params["ordered"] = bool(pd_dtype.ordered)
            params["categories_dtype"] = pd_dtype.categories.dtype.name

            # IMPORTANT: store integer codes, not values
            # -1 means missing
            array = series.cat.codes.to_numpy(copy=False)
        else:
            # for numeric-only SHM: require a real numpy array output here
            array = series.to_numpy(copy=False) if arr is None else arr

        # Optional param dtypes you mentioned (only if you support them)
        if isinstance(pd_dtype, pd.PeriodDtype):
            params["freq"] = str(pd_dtype.freq)

        if isinstance(pd_dtype, pd.IntervalDtype):
            params["subtype"] = np.dtype(pd_dtype.subtype).str
            params["closed"] = str(pd_dtype.closed)  # type: ignore[attr-defined]

        return {
            "name": series.name,
            "np_dtype": np.dtype(array.dtype).str,
            "pd_dtype": str(pd_dtype.name),
            "block": block,
            "params": params,
        }

    def _set_col_raw(self, col: str, value) -> None:
        # bypass our __setitem__
        pd.DataFrame.__setitem__(self, col, value)

    def _set_shm_metadata(self, meta: dict[str, ShmSeriesMetadata]):
        self._shm_metadata = meta

    @classmethod
    def _put_col_in_shm(
        cls, df: "ShmDataFrameColumns", col: str, shm_pool: Any, replace=False
    ):
        vals = df[col].to_numpy(copy=False)
        if (
            isinstance(vals, ShmArray)
            and hasattr(vals, "_shm_metadata")
            and not replace
        ):
            pass
        elif isinstance(vals, np.ndarray):
            arr_modified, ref, dtype, shape, order = ShmArray.to_shm(df[col], shm_pool)
            if arr_modified is not None:
                # Only needed for string-object normalization; for numeric/codes it's None
                df._set_col_raw(
                    col, pd.Series(arr_modified, index=df.index, name=col, copy=False)
                )
            df._shm_metadata[col] = cls._create_col_metadata(df[col], ref)
            return
        else:
            log_warning(
                f"Failed to update shm metadata for column '{col}' dtype {vals.dtype}"
            )
            return

    @classmethod
    def _reconstruct_series(
        cls, meta: ShmSeriesMetadata, block: ShmBlock, nrows: int
    ) -> pd.Series:
        col_name = meta["name"]
        col_name = str(col_name)
        dtype = np.dtype(meta["np_dtype"])
        pd_dtype = meta["pd_dtype"]
        spec = meta["block"]
        params = meta["params"]

        arr = ShmArray.from_block(block=block, shape=(nrows,), dtype=dtype, order="C")

        if pd_dtype == "category":
            cats = params.get("categories", None)
            ordered = bool(params.get("ordered", False))

            if cats is None:
                raise ValueError("category dtype requires params['categories']")

            cats_pd_dtype = params.get("categories_dtype")
            if cats_pd_dtype is not None:
                cats_index = pd.Index(cats, dtype=str(cats_pd_dtype))
            else:
                cats_index = pd.Index(cats)

            cat_dtype = pd.CategoricalDtype(categories=cats_index, ordered=ordered)

            # arr should hold integer codes
            # If arr holds codes: build categorical from codes without copying codes.
            # Pandas uses -1 for missing.
            cat = pd.Categorical.from_codes(cast(Sequence[int], arr), dtype=cat_dtype)
            return pd.Series(cat, name=col_name)

        # 2) tz-aware datetimes (only if you ever allow it)
        # Expect arr to be int64 ns timestamps
        if pd_dtype.startswith("datetime64[ns,") or pd_dtype == "datetime64[ns, tz]":
            tz = params.get("tz")
            if not tz:
                # if stored as a plain datetime64[ns] string, just fall through
                pass
            else:
                dt = pd.to_datetime(arr, unit="ns", utc=True).tz_convert(tz)
                return pd.Series(dt, name=col_name)

        return pd.Series(arr, name=col_name)

    @property
    def _constructor(self):
        return ShmDataFrameColumns

    def __setitem__(self, key, value):
        is_existing = key in self.columns

        super().__setitem__(key, value)

        if is_existing:
            self._on_column_replaced(key)
        else:
            self._on_column_added(key)

    def _on_column_added(self, col: str):
        from brmspy._singleton._shm_singleton import _get_shm

        shm = _get_shm()
        if not shm:
            return
        self._put_col_in_shm(self, col, shm)

    def _on_column_replaced(self, col: str):
        from brmspy._singleton._shm_singleton import _get_shm

        shm = _get_shm()
        if not shm:
            return
        self._put_col_in_shm(self, col, shm, replace=True)
