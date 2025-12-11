from typing import TYPE_CHECKING, Any, Callable, Optional

from ...types.session_types import SexpWrapper

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


# Pickle override
import copyreg


def _reduce_sexp(obj: "Sexp") -> tuple[Callable[..., Any], tuple[Any, ...]]:
    """
    Called by pickle whenever it encounters a Sexp instance.
    Must return (callable, args) or (callable, args, state, ...).
    """
    wrapper = cache_sexp(obj)
    # On unpickle: callable(*args) → object returned to the unpickler
    return (SexpWrapper, (wrapper._rid, wrapper._repr))


def _reduce_sexpwrapper(obj: SexpWrapper) -> tuple[Callable[..., Any], tuple[Any, ...]]:
    """
    Called by pickle whenever it encounters a Sexp instance.
    Must return (callable, args) or (callable, args, state, ...).
    """
    sexp = get_sexp(obj)
    # On unpickle: callable(*args) → object returned to the unpickler
    return (Sexp, (sexp,))


def register_global_pickle_overrides() -> None:
    copyreg.pickle(Sexp, _reduce_sexp)
    copyreg.pickle(SexpWrapper, _reduce_sexpwrapper)


# Make sure this runs before any pickling starts in the worker:
register_global_pickle_overrides()
