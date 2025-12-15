from collections.abc import Mapping
from typing import TYPE_CHECKING, Any, cast

from rpy2.rinterface_lib.sexp import NULL

from brmspy.types.shm_extensions import ShmArray

if TYPE_CHECKING:
    from rpy2.robjects import ListVector, Vector

import numpy as np
import rpy2.rinterface as rinterface

from brmspy.types.rpy2_converters import PyObject
from brmspy.types.shm import ShmPool

from rpy2.rinterface_lib.sexp import Sexp, StrSexpVector, RTYPES, SexpVector
from rpy2.rinterface import SexpVectorWithNumpyInterface


def _get_rvector_types(obj: Any) -> tuple[None | str, None | int]:
    if not isinstance(obj, SexpVectorWithNumpyInterface):
        return None, None

    dtypestr = obj._NP_TYPESTR
    itemsize = obj._R_SIZEOF_ELT

    if not dtypestr or not itemsize:
        return None, None

    return dtypestr, itemsize


def _get_rvector_memview(
    obj: Any,
) -> tuple[SexpVectorWithNumpyInterface | None, memoryview | None]:
    try:
        assert isinstance(obj, SexpVectorWithNumpyInterface) and isinstance(
            obj, SexpVector
        )  # assert types, shouldnt error by itself
        if hasattr(obj, "memoryview"):
            src = cast(Any, obj).memoryview()
            return obj, src
        else:
            return None, None
    except:
        return None, None


def _r2py_listvector(
    obj: "ListVector", shm: ShmPool | None = None
) -> dict[str, PyObject] | list[PyObject]:
    from ._dispatch import r_to_py

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


def _fallback_rvector_iter(obj):
    from rpy2.robjects.conversion import localconverter
    from rpy2.robjects import default_converter

    out = []
    with localconverter(default_converter) as cv:
        for el in obj:
            py = cv.rpy2py(el)
            out.append(py)
    return out


def _r2py_vector(obj: "Vector", shm: ShmPool | None = None) -> PyObject:
    import rpy2.robjects as ro
    from rpy2.robjects import default_converter
    from rpy2.robjects.conversion import localconverter

    assert not isinstance(obj, ro.ListVector)

    obj_any = cast(Any, obj)
    # length 1 → scalar
    if obj_any.__len__ and len(obj_any) == 1:
        # Try default R→Python conversion
        with localconverter(default_converter) as cv:
            py = cv.rpy2py(obj[0])
        return py

    dtypestr, itemsize = _get_rvector_types(obj)
    rvecnp, src = _get_rvector_memview(obj)

    # fallback
    if not dtypestr or not itemsize or not shm or not rvecnp or not src:
        return _fallback_rvector_iter(obj)

    # numpy convertible
    N = len(rvecnp)
    expected_bytes = itemsize * N
    dtype = np.dtype(dtypestr)

    if src.nbytes != expected_bytes:
        raise RuntimeError(f"R vector bytes={src.nbytes}, expected={expected_bytes}")

    # Allocate shm once
    block = shm.alloc(expected_bytes)
    assert block.shm.buf

    # Single bulk copy: R → shm, no intermediate ndarray
    src_bytes = src.cast("B")
    block.shm.buf[:expected_bytes] = src_bytes

    return ShmArray.from_block(block=block, shape=(N,), dtype=dtype)


def _py2r_list(obj: list | tuple) -> Sexp:
    import rpy2.robjects as ro
    from rpy2.robjects import default_converter, numpy2ri, pandas2ri
    from rpy2.robjects.conversion import localconverter

    if not obj:
        return ro.ListVector({})

    if isinstance(obj, tuple):
        obj = list(obj)

    from ._dispatch import py_to_r

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
