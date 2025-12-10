from typing import Any
from brmspy.session.transport import ShmBlock
import numpy as np
import pandas as pd


class ShmArray(np.ndarray):
    block: ShmBlock  # for type checkers

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
        obj.block = block
        return obj


class ShmDataFrameColumns(pd.DataFrame):
    blocks_columns: dict[str, ShmBlock]

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
            arr.block = block
            _data[column] = arr

        df = ShmDataFrameColumns(data=_data, index=index)
        df.blocks_columns = arrays
        return df
