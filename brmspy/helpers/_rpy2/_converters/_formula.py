from typing import Callable, cast, get_args

from rpy2.rinterface_lib.sexp import Sexp
from brmspy.types.formula_dsl import FormulaPart, _FORMULA_FUNCTION_WHITELIST

from brmspy.types.formula_dsl import FormulaPart

from ._dispatch import py_to_r


def _py2r_formula_part(obj: FormulaPart) -> Sexp:
    import rpy2.robjects as ro

    args = [py_to_r(o) for o in obj._args]
    kwargs = {k: py_to_r(v) for k, v in obj._kwargs.items()}

    assert obj._fun in get_args(_FORMULA_FUNCTION_WHITELIST)

    fun = cast(Callable, ro.r(f"brms::{obj._fun}"))
    return fun(*args, **kwargs)
