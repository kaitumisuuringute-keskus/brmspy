from typing import Any

import numpy as np
import pandas as pd

from brmspy.types.shm import ShmBlock, ShmBlockSpec


class ShmArray(np.ndarray):
    block: ShmBlockSpec  # for type checkers

    @classmethod
    def from_block(
        cls, block: ShmBlock, shape: tuple[int, ...], dtype: np.dtype, **kwargs
    ) -> "ShmArray":
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
        _dtype = np.dtype(dtype)
        arr = ShmArray.from_block(shape=(ncols, nrows), dtype=_dtype, block=block)

        df = ShmDataFrameSimple(data=arr.T, index=index, columns=columns)
        df.block = ShmBlockSpec(name=block.name, size=block.size)
        return df


class ShmDataFrameColumns(pd.DataFrame):
    blocks_columns: dict[str, ShmBlockSpec]

    @classmethod
    def from_blocks(
        cls, arrays: dict[str, ShmBlock], dtypes: dict[str, str], index: list[Any]
    ) -> "ShmDataFrameColumns":
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
