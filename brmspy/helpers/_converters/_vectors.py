from collections.abc import Mapping
import os
from typing import Any, Dict, Union, List, cast
from rpy2.rinterface_lib.sexp import NULL
import rpy2.robjects as ro
from rpy2.robjects import default_converter, pandas2ri, numpy2ri
from rpy2.robjects.conversion import localconverter

from brmspy.helpers._converters._converter_types import PyObject
from brmspy.session.transport import ShmPool
import numpy as np


def _r2py_listvector(
    obj: ro.ListVector, shm: ShmPool | None = None
) -> dict[str, PyObject] | list[PyObject]:
    from ._registry import r_to_py

    names = list(obj.names) if obj.names is not NULL else None

    # Named list → dict
    if names and any(n is not NULL and n != "" for n in names):
        result = {}
        for name in names:
            key = str(name) if name not in (None, "") else None
            if obj is NULL:
                result[key] = None
            else:
                result[key] = r_to_py(obj.rx2(name))
        return result

    # Unnamed → list
    return [r_to_py(el) for el in obj]


def _r2py_vector(obj: ro.vectors.Vector, shm: ShmPool | None = None) -> PyObject:
    assert not isinstance(obj, ro.ListVector)

    obj_any = cast(Any, obj)
    # length 1 → scalar
    if obj_any.__len__ and len(obj_any) == 1:
        # Try default R→Python conversion
        with localconverter(default_converter) as cv:
            py = cv.rpy2py(obj[0])
        return py

    # length >1 → list of scalars
    out = []
    for el in obj:
        with localconverter(default_converter) as cv:
            py = cv.rpy2py(el)
        out.append(py)
    return out


def _py2r_list(obj: list | tuple) -> ro.Sexp:
    if not obj:
        return ro.ListVector({})

    from ._registry import py_to_r

    if all(isinstance(el, Mapping) for el in obj):
        # R lists are usually named or indexed; use 1-based index names
        converted = {str(i + 1): py_to_r(el) for i, el in enumerate(obj)}
        return ro.ListVector(converted)

    # Homogeneous scalar lists → atomic R vectors (c(...))
    # Strings
    if all(isinstance(el, str) for el in obj):
        return ro.StrVector(list(obj))

    # Booleans
    if all(isinstance(el, bool) for el in obj):
        return ro.BoolVector(list(obj))

    # Integers (avoid treating bools as ints)
    if all(
        isinstance(el, (int, np.integer)) and not isinstance(el, bool) for el in obj
    ):
        return ro.IntVector(list(obj))

    # Numeric (mix of ints/floats/bools) → R "numeric" (double) vector
    if all(isinstance(el, (int, float, np.integer, np.floating, bool)) for el in obj):
        return ro.FloatVector([float(el) for el in obj])

    # mixed / other lists: let rpy2 decide (vectors, lists, etc.)
    with localconverter(
        default_converter + pandas2ri.converter + numpy2ri.converter
    ) as cv:
        return cv.py2rpy(obj)
