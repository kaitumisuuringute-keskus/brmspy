from typing import TYPE_CHECKING, Any, Callable, Optional

from ...types.session_types import SexpWrapper

from rpy2.rinterface_lib.sexp import Sexp
from rpy2.rinterface_lib.sexp import NULL


_SEXP_CACHE: dict[int, Sexp] = {}


def get_sexp(rid: int) -> Sexp:
    if rid in _SEXP_CACHE:
        return _SEXP_CACHE[rid]
    return NULL


def _cache_single(obj: Sexp) -> SexpWrapper:
    _SEXP_CACHE[obj.rid] = obj
    return SexpWrapper(_rid=obj.rid, _repr=str(obj))


def cache_sexp(obj: Any) -> Any:
    if isinstance(obj, Sexp):
        return _cache_single(obj)
    elif hasattr(obj, "r") and isinstance(obj.r, Sexp):
        obj.r = _cache_single(obj.r)
        return obj
    elif isinstance(obj, list):
        return [cache_sexp(o) for o in obj]
    elif isinstance(obj, dict):
        return {k: cache_sexp(v) for k, v in obj.items()}
    return obj


def _reattach_sexp_single(obj: Any) -> Any:

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


def reattach_sexp(obj: Any) -> Any:
    if isinstance(obj, list):
        return [reattach_sexp(v) for v in obj]
    elif isinstance(obj, dict):
        return {k: reattach_sexp(v) for k, v in obj.items()}
    elif hasattr(obj, "r") and isinstance(obj.r, SexpWrapper):
        if obj.r._rid in _SEXP_CACHE:
            obj.r = _SEXP_CACHE[obj.r._rid]
        else:
            obj.r = None
    elif isinstance(obj, SexpWrapper):
        if obj._rid in _SEXP_CACHE:
            return _SEXP_CACHE[obj._rid]
        else:
            return None
    return obj


# Pickle override
import copyreg


def _reduce_sexp(obj: Sexp) -> tuple[Callable[..., Any], tuple[Any, ...]]:
    """
    Called by pickle whenever it encounters a Sexp instance.
    Must return (callable, args) or (callable, args, state, ...).
    """
    wrapper = _cache_single(obj)
    # On unpickle: callable(*args) → object returned to the unpickler
    return (SexpWrapper, (wrapper._rid, wrapper._repr))


def _reduce_sexpwrapper(obj: SexpWrapper) -> tuple[Callable[..., Any], tuple[Any, ...]]:
    """
    Called by pickle whenever it encounters a Sexp instance.
    Must return (callable, args) or (callable, args, state, ...).
    """
    # On unpickle: callable(*args) → object returned to the unpickler
    return (get_sexp, (obj._rid,))


def register_global_pickle_overrides() -> None:
    copyreg.pickle(Sexp, _reduce_sexp)
    copyreg.pickle(SexpWrapper, _reduce_sexpwrapper)


# Make sure this runs before any pickling starts in the worker:
register_global_pickle_overrides()
