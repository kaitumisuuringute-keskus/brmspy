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
    ShmSeriesMetadata,
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
    if "_obs_id_" in df.columns and not df["_obs_id_"].duplicated().any():
        df.index = df["_obs_id_"]
    elif "obs_id" in df.columns and not df["obs_id"].duplicated().any():
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

    if "_obs_id_" in df.columns and not df["_obs_id_"].duplicated().any():
        df.index = df["_obs_id_"]
    elif "obs_id" in df.columns and not df["obs_id"].duplicated().any():
        df.index = df["obs_id"]

    return df


# CONVERTERS


def _r2py_matrix(obj: "Matrix", shm: ShmPool | None = None) -> PyObject:
    return _rmatrix_to_py(obj=obj, shm=shm)


def _r2py_dataframe_fallback(obj: "DataFrame") -> PyObject:
    """
    Fallback conversion for R data.frame -> pandas.DataFrame.

    Notes
    -----
    In some environments, rpy2/pandas2ri may convert R factors to their underlying
    integer codes instead of `pandas.Categorical`. Since brmspy relies on factors
    roundtripping as categoricals (and we have custom factor handling in
    [`_r2py_vector()`][brmspy.helpers._rpy2._converters._vectors._r2py_vector]),
    we patch factor columns explicitly here.
    """
    import rpy2.robjects as ro
    from rpy2.robjects import pandas2ri
    from rpy2.robjects.conversion import localconverter

    with localconverter(pandas2ri.converter) as cv:
        df = cv.rpy2py(obj)

    # Ensure factor columns come back as pandas categoricals.
    # (Otherwise they can appear as int32 codes with NA_INTEGER sentinel values.)
    try:
        if obj.names is not NULL:
            for name in list(obj.names):
                col_name = str(name)
                if col_name not in df.columns:
                    continue

                col_r = obj.rx2(name)
                if isinstance(col_r, ro.FactorVector):
                    cat = _r2py_vector(col_r, shm=None, allow_scalar=False)
                    df[col_name] = pd.Series(
                        cast(np.ndarray, cat), index=df.index, name=col_name, copy=False
                    )
    except Exception:
        # Best-effort: never let fallback conversion fail due to factor patching.
        pass

    return _adjust_df_for_py(df)


def _r2py_dataframe(obj: "DataFrame", shm: ShmPool | None = None) -> PyObject:
    if not shm:
        return _r2py_dataframe_fallback(obj)

    try:
        from rpy2.robjects.pandas2ri import _flatten_dataframe

        rownames = list(obj.rownames)
        if len(rownames) == 0:
            return pd.DataFrame({})

        # convert straight into ShmDataframeColumns
        colnames_lst = []
        od = OrderedDict()
        od_r = OrderedDict()
        for i, col in enumerate(_flatten_dataframe(obj, colnames_lst)):
            arr = _r2py_vector(col, shm, allow_scalar=False)
            od_r[i] = col
            od[i] = arr

        res = ShmDataFrameColumns(od)
        res.columns = tuple(
            ".".join(_) if isinstance(_, list) else _ for _ in colnames_lst
        )
        res.index = rownames
        res._set_shm_metadata({})

        if res.empty:
            return res

        for col in res.columns:
            # Side-effect: ShmDataFrameColumns overrides __setitem__/events;
            # assigning the same Series forces the "put column to SHM" path
            # DO NOT REMOVE!
            res[col] = res[col]
        res = _adjust_df_for_py(res)

        return res
    except Exception as e:
        log_warning(
            f"ShmDataFrameColumns conversion failed, falling back to default. Reason: {e}"
        )
        return _r2py_dataframe_fallback(obj)


def _adjust_df_for_r(obj: pd.DataFrame) -> pd.DataFrame:
    if "_obs_id_" not in obj.columns:
        if obj.index.nlevels == 1:
            obj["_obs_id_"] = obj.index
        else:
            obj = obj.assign(_obs_id_=range(len(obj)))

    for c in obj.columns:
        s = obj[c]
        if isinstance(s.dtype, pd.CategoricalDtype):
            cats = s.dtype.categories
            # needs string categories for factor conversion
            if cats.inferred_type != "string":
                # log_warning(
                #    f"Column {c} has non-string categories, converting to string."
                # )
                obj[c] = s.cat.rename_categories(cats.map(str))

        elif pd.api.types.is_integer_dtype(s.dtype) and s.dtype != np.int32:
            obj[c] = obj[c].astype("int32")
        elif pd.api.types.is_float_dtype(s.dtype) and s.dtype != np.float64:
            obj[c] = obj[c].astype("float64")

    return obj


def _adjust_df_for_py(df: pd.DataFrame) -> pd.DataFrame:
    if "_obs_id_" in df.columns:
        df = df.set_index("_obs_id_", drop=True)
    return df


def _py2r_dataframe(obj: pd.DataFrame) -> Sexp:
    from rpy2.robjects import pandas2ri
    from rpy2.robjects.conversion import localconverter

    obj = _adjust_df_for_r(obj)

    with localconverter(pandas2ri.converter) as cv:
        return cv.py2rpy(obj)


def _py2r_numpy(obj: np.ndarray) -> Sexp:
    from rpy2.robjects import numpy2ri, pandas2ri
    from rpy2.robjects.conversion import localconverter

    with localconverter(pandas2ri.converter + numpy2ri.converter) as cv:
        return cv.py2rpy(obj)
