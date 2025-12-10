from collections.abc import Mapping
from typing import TYPE_CHECKING, Union

if TYPE_CHECKING:
    import rpy2.robjects as ro


from brmspy.helpers._converters._converter_types import PyObject
from brmspy.session.transport import ShmPool


def _r2py_fallback(obj: ro.Sexp, shm: ShmPool | None = None) -> PyObject:
    from rpy2.robjects import default_converter
    from rpy2.robjects.conversion import localconverter

    try:
        with localconverter(default_converter) as cv:
            return cv.rpy2py(obj)
    except Exception:
        return str(obj)


def _py2r_fallback(obj: PyObject) -> ro.Sexp:
    from rpy2.robjects import default_converter, pandas2ri, numpy2ri
    from rpy2.robjects.conversion import localconverter

    with localconverter(
        default_converter + pandas2ri.converter + numpy2ri.converter
    ) as cv:
        return cv.py2rpy(obj)


def _r2py_language(
    obj: Union[ro.Formula, ro.language.LangVector, ro.SignatureTranslatedFunction],
    shm: ShmPool | None = None,
) -> PyObject:
    return str(obj)


def _py2r_mapping(
    obj: Mapping,
) -> ro.Sexp:
    from ._registry import py_to_r

    converted = {str(k): py_to_r(v) for k, v in obj.items()}
    return ro.ListVector(converted)
