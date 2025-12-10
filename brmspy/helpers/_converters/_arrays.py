import os
import rpy2.robjects as ro
import pandas as pd
import numpy as np
from rpy2.robjects import pandas2ri, numpy2ri
from rpy2.robjects.conversion import localconverter
import rpy2.robjects as ro
from rpy2.rinterface_lib import sexp as r_sexp
from ._converter_types import PyObject
from brmspy.session.transport import ShmPool


# HELPERS


# CONVERTERS


def _r2py_matrix(obj: ro.Matrix, shm: ShmPool | None = None) -> PyObject:
    if len(obj.dim) != 2:
        raise Exception("Matrix with dims != 2. Unimplemented conversion")
    if obj.colnames != ro.NULL:
        colnames = [str(el) for el in obj.colnames]
    else:
        colnames = None
    if obj.rownames != ro.NULL:
        rownames = [str(el) for el in obj.rownames]
    else:
        rownames = None

    if rownames is None and colnames is None:
        return np.asarray(obj)
    else:
        return pd.DataFrame(data=np.asarray(obj), columns=colnames, index=rownames)


def _r2py_dataframe(obj: ro.DataFrame, shm: ShmPool | None = None) -> PyObject:
    with localconverter(pandas2ri.converter) as cv:
        return cv.rpy2py(obj)


def _py2r_dataframe(obj: pd.DataFrame) -> ro.Sexp:
    with localconverter(pandas2ri.converter) as cv:
        return cv.py2rpy(obj)


def _py2r_numpy(obj: np.ndarray) -> ro.Sexp:
    with localconverter(pandas2ri.converter + numpy2ri.converter) as cv:
        return cv.py2rpy(obj)
