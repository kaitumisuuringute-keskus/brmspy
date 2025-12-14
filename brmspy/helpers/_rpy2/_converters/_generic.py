from collections.abc import Mapping
from typing import TYPE_CHECKING, Union

if TYPE_CHECKING:
    from rpy2.robjects import Formula, SignatureTranslatedFunction

from rpy2.rinterface import LangSexpVector
from rpy2.rinterface_lib.sexp import Sexp

from brmspy.types.rpy2_converters import PyObject
from brmspy.types.shm import ShmPool

from ._dispatch import py_to_r


def _r2py_fallback(obj: Sexp, shm: ShmPool | None = None) -> PyObject:
    from rpy2.robjects import default_converter
    from rpy2.robjects.conversion import localconverter

    try:
        with localconverter(default_converter) as cv:
            return cv.rpy2py(obj)
    except Exception:
        return str(obj)


def _py2r_fallback(obj: PyObject) -> Sexp:
    from rpy2.robjects import default_converter, numpy2ri, pandas2ri
    from rpy2.robjects.conversion import localconverter

    with localconverter(
        default_converter + pandas2ri.converter + numpy2ri.converter
    ) as cv:
        return cv.py2rpy(obj)


def _r2py_language(
    obj: Union["Formula", "LangSexpVector", "SignatureTranslatedFunction"],
    shm: ShmPool | None = None,
) -> PyObject:
    return str(obj)


def _py2r_mapping(
    obj: Mapping,
) -> Sexp:
    import rpy2.robjects as ro

    converted = {str(k): py_to_r(v) for k, v in obj.items()}
    return ro.ListVector(converted)
