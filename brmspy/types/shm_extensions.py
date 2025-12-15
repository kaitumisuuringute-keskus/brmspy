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
from typing import Any, TypedDict, Union

import numpy as np
import pandas as pd

from brmspy.types.shm import ShmBlock, ShmRef

__all__ = ["ShmArray", "ShmDataFrameSimple", "ShmDataFrameColumns"]


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

    block: ShmRef  # for type checkers

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
            obj.block = {
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


class ShmDataFrameSimple(pd.DataFrame):
    """
    pandas DataFrame backed by a single shared-memory block (numeric only).

    Attributes
    ----------
    block : ShmRef
        Reference to the shared-memory block backing the DataFrame's values.
    """

    block: ShmRef

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
        df.block = {
            "name": block.name,
            "size": block.size,
            "content_size": block.content_size,
        }
        return df


class PandasColumnMetadata(TypedDict):
    name: str
    np_dtype: str
    pd_type: str
    params: dict[str, Any]  # extra info per logical type
    block: ShmRef


class ShmDataFrameColumns(pd.DataFrame):
    """
    pandas DataFrame backed by per-column shared-memory blocks (numeric only).

    Attributes
    ----------
    blocks_columns : dict[str, PandasColumnMetadata]
        Mapping from column name to data required for its reconstruction
    """

    blocks_columns: dict[str, PandasColumnMetadata]
