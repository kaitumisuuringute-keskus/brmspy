from collections import OrderedDict
from typing import TYPE_CHECKING, Any, cast

import numpy as np
import pandas as pd

from brmspy.helpers._rpy2._converters._vectors import (
    _get_rvector_memview,
    _get_rvector_types,
    _r2py_vector,
)
from brmspy.helpers.log import log_warning
from brmspy.types.shm import ShmPool
from brmspy.types.shm_extensions import (
    PandasColumnMetadata,
    ShmArray,
    ShmDataFrameColumns,
    ShmDataFrameSimple,
)

from ....types.rpy2_converters import PyObject

if TYPE_CHECKING:
    from rpy2.robjects import DataFrame, Matrix

from rpy2.rinterface import SexpVectorWithNumpyInterface
from rpy2.rinterface_lib.sexp import NULL, Sexp, SexpVector

# HELPERS


def _rmatrix_info(obj: "Matrix") -> tuple[int, int, list[str] | None, list[str] | None]:
    nrow, ncol = obj.dim

    if obj.colnames != NULL:
        colnames = [str(el) for el in obj.colnames]
    else:
        colnames = None
    if obj.rownames != NULL:
        rownames = [str(el) for el in obj.rownames]
    else:
        rownames = None

    return nrow, ncol, rownames, colnames


def _rmatrix_to_py_default(obj: "Matrix") -> pd.DataFrame | np.ndarray:
    nrow, ncol, rownames, colnames = _rmatrix_info(obj)

    if not rownames and not colnames:
        return np.array(obj)

    df = pd.DataFrame(data=np.array(obj), columns=colnames, index=rownames)
    if "obs_id" in df.columns and not df["obs_id"].duplicated().any():
        df.index = df["obs_id"]
    return df


def _rmatrix_to_py(
    obj: "Matrix", shm: ShmPool | None = None
) -> pd.DataFrame | np.ndarray | ShmArray | ShmDataFrameSimple:
    if len(obj.dim) != 2:
        raise Exception("Matrix with dims != 2. Unimplemented conversion")

    # No shm, fall back to regular numpy
    if shm is None:
        return np.array(obj)

    dtypestr, itemsize = _get_rvector_types(obj)

    if not dtypestr or not itemsize:
        return _rmatrix_to_py_default(obj)

    dtype = np.dtype(dtypestr)

    rvecnp, src = _get_rvector_memview(obj)
    if rvecnp is None or src is None:
        return _rmatrix_to_py_default(obj)

    nrow, ncol, rownames, colnames = _rmatrix_info(obj)

    expected_bytes = nrow * ncol * itemsize

    # Raw buffer view over R's underlying data (column-major)
    if src.nbytes != expected_bytes:
        raise RuntimeError(f"R matrix bytes={src.nbytes}, expected={expected_bytes}")

    # Allocate shm once
    block = shm.alloc(expected_bytes)
    assert block.shm.buf

    # Single bulk copy: R → shm, no intermediate ndarray
    src_bytes = src.cast("B")
    block.shm.buf[:expected_bytes] = src_bytes

    # Wrap shm buffer as a numpy array, matching R's column-major layout
    if not rownames and not colnames:
        return ShmArray.from_block(block=block, shape=(nrow, ncol), dtype=dtype)

    df = ShmDataFrameSimple.from_block(
        block=block,
        nrows=nrow,
        ncols=ncol,
        columns=colnames,
        index=rownames,
        dtype=dtype,
    )

    if "obs_id" in df.columns and not df["obs_id"].duplicated().any():
        df.index = df["obs_id"]

    return df


# CONVERTERS


def _r2py_matrix(obj: "Matrix", shm: ShmPool | None = None) -> PyObject:
    return _rmatrix_to_py(obj=obj, shm=shm)


def _r2py_dataframe_fallback(obj: "DataFrame") -> PyObject:
    from rpy2.robjects import pandas2ri
    from rpy2.robjects.conversion import localconverter

    with localconverter(pandas2ri.converter) as cv:
        return cv.rpy2py(obj)


def _r2py_dataframe(obj: "DataFrame", shm: ShmPool | None = None) -> PyObject:
    if not shm:
        return _r2py_dataframe_fallback(obj)

    try:
        from rpy2.robjects.pandas2ri import _flatten_dataframe

        # convert straight into ShmDataframeColumns
        colnames_lst = []
        od = OrderedDict()
        od_r = OrderedDict()
        for i, col in enumerate(_flatten_dataframe(obj, colnames_lst)):
            arr = _r2py_vector(col, shm)
            od_r[i] = col
            od[i] = arr

        res = ShmDataFrameColumns(od)
        res.columns = tuple(
            ".".join(_) if isinstance(_, list) else _ for _ in colnames_lst
        )
        res.index = obj.rownames

        cols_metadata: dict[str, PandasColumnMetadata] = {}
        for idx, col in enumerate(res.columns):
            arr = od[idx]
            arr_r = od_r[idx]

            if isinstance(arr, list):
                arr = np.array(arr)
                block, dtype, _, _ = ShmArray.to_shm(arr, shm)
            elif isinstance(arr, ShmArray):
                dtype = str(arr.dtype)
                block = arr.block
            else:
                raise Exception(
                    f"{col} is not a ShmArray, found {type(arr)} and R type {type(arr_r)}"
                )

            cols_metadata[col] = {
                "name": col,
                "np_dtype": dtype,
                "pd_type": dtype,
                "block": block,
                "params": {},
            }

        res._blocks_columns = cols_metadata

        return res
    except Exception as e:
        log_warning(
            f"ShmDataFrameColumns conversion failed, falling back to default. Reason: {e}"
        )
        return _r2py_dataframe_fallback(obj)


def _py2r_dataframe(obj: pd.DataFrame) -> Sexp:
    from rpy2.robjects import pandas2ri
    from rpy2.robjects.conversion import localconverter

    # if "obs_id" not in obj.columns:
    #    obj = obj.reset_index(drop=False, names="obs_id")

    with localconverter(pandas2ri.converter) as cv:
        return cv.py2rpy(obj)


def _py2r_numpy(obj: np.ndarray) -> Sexp:
    from rpy2.robjects import numpy2ri, pandas2ri
    from rpy2.robjects.conversion import localconverter

    with localconverter(pandas2ri.converter + numpy2ri.converter) as cv:
        return cv.py2rpy(obj)
