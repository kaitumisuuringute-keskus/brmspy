from typing import TYPE_CHECKING
import pandas as pd
import numpy as np
from ._converter_types import PyObject
from brmspy.session.transport import ShmPool

if TYPE_CHECKING:
    from rpy2.robjects import DataFrame, Matrix

from rpy2.rinterface_lib.sexp import Sexp, NULL


# HELPERS


# CONVERTERS


def _r2py_matrix(obj: Matrix, shm: ShmPool | None = None) -> PyObject:
    if len(obj.dim) != 2:
        raise Exception("Matrix with dims != 2. Unimplemented conversion")
    if obj.colnames != NULL:
        colnames = [str(el) for el in obj.colnames]
    else:
        colnames = None
    if obj.rownames != NULL:
        rownames = [str(el) for el in obj.rownames]
    else:
        rownames = None

    if rownames is None and colnames is None:
        return np.asarray(obj)
    else:
        return pd.DataFrame(data=np.asarray(obj), columns=colnames, index=rownames)


def _r2py_dataframe(obj: DataFrame, shm: ShmPool | None = None) -> PyObject:
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
    from rpy2.robjects import pandas2ri, numpy2ri
    from rpy2.robjects.conversion import localconverter

    with localconverter(pandas2ri.converter + numpy2ri.converter) as cv:
        return cv.py2rpy(obj)
