from typing import TYPE_CHECKING, Any, cast

import numpy as np
import pandas as pd

from brmspy.types.shm import ShmPool
from brmspy.types.shm_extensions import ShmArray, ShmDataFrameSimple

from ....types.rpy2_converters import PyObject

if TYPE_CHECKING:
    from rpy2.robjects import DataFrame, Matrix

from rpy2.rinterface import SexpVectorWithNumpyInterface
from rpy2.rinterface_lib.sexp import NULL, Sexp, SexpVector

# HELPERS


def _get_vector_types(obj: Any) -> tuple[None | str, None | int]:
    if not isinstance(obj, SexpVectorWithNumpyInterface):
        return None, None

    dtypestr = obj._NP_TYPESTR
    itemsize = obj._R_SIZEOF_ELT

    if not dtypestr or not itemsize:
        return None, None

    return dtypestr, itemsize


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

    return pd.DataFrame(data=np.array(obj), columns=colnames, index=rownames)


def _rmatrix_to_py(
    obj: "Matrix", shm: ShmPool | None = None
) -> pd.DataFrame | np.ndarray | ShmArray | ShmDataFrameSimple:
    if len(obj.dim) != 2:
        raise Exception("Matrix with dims != 2. Unimplemented conversion")

    # No shm, fall back to regular numpy
    if shm is None:
        return np.array(obj)

    dtypestr, itemsize = _get_vector_types(obj)

    if not dtypestr or not itemsize:
        return _rmatrix_to_py_default(obj)

    dtype = np.dtype(dtypestr)

    assert isinstance(obj, SexpVectorWithNumpyInterface) and isinstance(
        obj, SexpVector
    )  # assert types, shouldnt error by itself
    if hasattr(obj, "memoryview"):
        src = cast(Any, obj).memoryview()
    else:
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

    return ShmDataFrameSimple.from_block(
        block=block,
        nrows=nrow,
        ncols=ncol,
        columns=colnames,
        index=rownames,
        dtype=dtype,
    )


# CONVERTERS


def _r2py_matrix(obj: "Matrix", shm: ShmPool | None = None) -> PyObject:
    return _rmatrix_to_py(obj=obj, shm=shm)


def _r2py_dataframe(obj: "DataFrame", shm: ShmPool | None = None) -> PyObject:
    from rpy2.robjects import pandas2ri
    from rpy2.robjects.conversion import localconverter

    with localconverter(pandas2ri.converter) as cv:
        return cv.rpy2py(obj)


def _py2r_dataframe(obj: pd.DataFrame) -> Sexp:
    from rpy2.robjects import pandas2ri
    from rpy2.robjects.conversion import localconverter

    with localconverter(pandas2ri.converter) as cv:
        return cv.py2rpy(obj)


def _py2r_numpy(obj: np.ndarray) -> Sexp:
    from rpy2.robjects import numpy2ri, pandas2ri
    from rpy2.robjects.conversion import localconverter

    with localconverter(pandas2ri.converter + numpy2ri.converter) as cv:
        return cv.py2rpy(obj)
