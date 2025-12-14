from typing import TYPE_CHECKING, Any, Optional

from ...types.session import SexpWrapper

if TYPE_CHECKING:
    from rpy2.rinterface_lib.sexp import Sexp


_SEXP_CACHE: dict[int, "Sexp"] = {}


def get_sexp(ref: SexpWrapper) -> Optional["Sexp"]:
    from rpy2.rinterface_lib.sexp import NULL

    if ref._rid in _SEXP_CACHE:
        return _SEXP_CACHE[ref._rid]
    return NULL


def cache_sexp(obj: "Sexp") -> SexpWrapper:
    _SEXP_CACHE[obj.rid] = obj
    return SexpWrapper(_rid=obj.rid, _repr=str(obj))


def reattach_sexp(obj: Any) -> Any:

    if hasattr(obj, "r") and isinstance(obj.r, SexpWrapper):
        if obj.r._rid in _SEXP_CACHE:
            obj.r = _SEXP_CACHE[obj.r._rid]
        else:
            obj.r = None
    if isinstance(obj, SexpWrapper):
        if obj._rid in _SEXP_CACHE:
            return _SEXP_CACHE[obj._rid]
        else:
            return None
    return obj
