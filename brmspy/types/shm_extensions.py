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

from typing import Any

import numpy as np
import pandas as pd

from brmspy.types.shm import ShmBlock, ShmBlockSpec

__all__ = ["ShmArray", "ShmDataFrameSimple", "ShmDataFrameColumns"]


class ShmArray(np.ndarray):
    """
    NumPy array view backed by a shared-memory block.

    Attributes
    ----------
    block : ShmBlockSpec
        Reference to the shared-memory block backing the array data.

    Notes
    -----
    This is a *view* over `SharedMemory.buf`. Closing/unlinking the underlying
    shared memory while the array is still in use will lead to undefined
    behavior.
    """

    block: ShmBlockSpec  # for type checkers

    @classmethod
    def from_block(
        cls, block: ShmBlock, shape: tuple[int, ...], dtype: np.dtype, **kwargs
    ) -> "ShmArray":
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
        base = np.ndarray(
            shape=shape,
            dtype=dtype,
            buffer=block.shm.buf,
            order="F",
        )
        obj = base.view(ShmArray)
        obj.block = ShmBlockSpec(name=block.name, size=block.size)
        return obj


class ShmDataFrameSimple(pd.DataFrame):
    """
    pandas DataFrame backed by a single shared-memory block (numeric only).

    Attributes
    ----------
    block : ShmBlockSpec
        Reference to the shared-memory block backing the DataFrame's values.
    """

    block: ShmBlockSpec

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
        df.block = ShmBlockSpec(name=block.name, size=block.size)
        return df


class ShmDataFrameColumns(pd.DataFrame):
    """
    pandas DataFrame backed by per-column shared-memory blocks (numeric only).

    Attributes
    ----------
    blocks_columns : dict[str, ShmBlockSpec]
        Mapping from column name to its shared-memory block reference.
    """

    blocks_columns: dict[str, ShmBlockSpec]

    @classmethod
    def from_blocks(
        cls, arrays: dict[str, ShmBlock], dtypes: dict[str, str], index: list[Any]
    ) -> "ShmDataFrameColumns":
        """
        Construct a DataFrame backed by one SHM block per column.

        Parameters
        ----------
        arrays : dict[str, ShmBlock]
            Mapping from column name to attached shared-memory block containing
            that column's 1D values.
        dtypes : dict[str, str]
            Mapping from column name to dtype string.
        index : list[Any]
            Index labels.

        Returns
        -------
        ShmDataFrameColumns
        """
        _data: dict[str, ShmArray] = {}

        length = len(index)

        for column, block in arrays.items():
            dtype = np.dtype(dtypes[column])
            arr = ShmArray(
                shape=(length,),
                dtype=dtype,
                buffer=block.shm.buf,
            )
            arr.block = ShmBlockSpec(block.name, block.size)
            _data[column] = arr

        df = ShmDataFrameColumns(data=_data, index=index)
        df.blocks_columns = {k: ShmBlockSpec(v.name, v.size) for k, v in arrays.items()}
        return df
