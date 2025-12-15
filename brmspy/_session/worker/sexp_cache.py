"""
Worker-side cache for rpy2 `Sexp` objects (internal).

The main process must not hold live rpy2 objects. Instead, the worker replaces R
objects with lightweight [`SexpWrapper`][brmspy.types.session.SexpWrapper] handles and
stores the real `Sexp` in a local cache keyed by `rid`.

This module also installs pickle reducers so that any accidental pickling of a `Sexp`
turns into a wrapper rather than attempting to serialize the R object.
"""

from typing import Any, Callable

from rpy2.rinterface_lib.sexp import NULL, Sexp

from ...types.session import SexpWrapper

_SEXP_CACHE: dict[int, Sexp] = {}


def get_sexp(rid: int) -> Sexp:
    """
    Fetch a cached `Sexp` by rid.

    Returns `NULL` when the rid is not present.
    """
    if rid in _SEXP_CACHE:
        return _SEXP_CACHE[rid]
    return NULL


def _cache_single(obj: Sexp) -> SexpWrapper:
    """Store `obj` in the cache and return a lightweight wrapper for IPC."""
    _SEXP_CACHE[obj.rid] = obj
    _repr = str(obj)
    if len(_repr) > 16384:
        _repr = _repr[:16384]
    return SexpWrapper(_rid=obj.rid, _repr=_repr)


def _extract_sexp(o: Any) -> Sexp | None:
    # Fast path: already a low-level Sexp
    if isinstance(o, Sexp):
        return o

    # robjects wrappers (Vector/Matrix/etc.) are not instances of Sexp,
    # but usually expose the underlying Sexp via __sexp__ or _sexp.
    sexp = getattr(o, "__sexp__", None)
    if isinstance(sexp, Sexp):
        return sexp

    sexp = getattr(o, "_sexp", None)
    if isinstance(sexp, Sexp):
        return sexp

    return None


def cache_sexp(obj: Any) -> Any:
    """
    Replace any embedded-R objects inside `obj` with `SexpWrapper` handles.

    Supports:
    - plain `rpy2.rinterface_lib.sexp.Sexp`
    - rpy2.robjects wrappers (e.g. vectors/matrices), by extracting the underlying Sexp
    - objects with an `.r` attribute
    - list/dict containers (recursively)

    This keeps the main process free of rpy2/embedded-R objects.
    """

    sexp = _extract_sexp(obj)
    if sexp is not None:
        return _cache_single(sexp)

    if hasattr(obj, "r"):
        obj.r = cache_sexp(obj.r)

    if isinstance(obj, list):
        return [cache_sexp(o) for o in obj]
    if isinstance(obj, dict):
        return {k: cache_sexp(v) for k, v in obj.items()}

    return obj


def reattach_sexp(obj: Any) -> Any:
    """
    Replace any `SexpWrapper` handles inside `obj` with the cached `Sexp`.

    If a wrapper cannot be resolved (rid not in cache), the wrapper is replaced
    with `None`.
    """
    if isinstance(obj, list):
        return [reattach_sexp(v) for v in obj]
    elif isinstance(obj, dict):
        return {k: reattach_sexp(v) for k, v in obj.items()}
    elif hasattr(obj, "r"):
        obj.r = reattach_sexp(obj.r)
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
    Pickle reducer for `Sexp` (worker-side).

    Converts the `Sexp` into a cached [`SexpWrapper`][brmspy.types.session.SexpWrapper] so
    the main process never receives a live rpy2 object.
    """
    wrapper = _cache_single(obj)
    return (SexpWrapper, (wrapper._rid, wrapper._repr))


def _reduce_sexpwrapper(obj: SexpWrapper) -> tuple[Callable[..., Any], tuple[Any, ...]]:
    """
    Pickle reducer for `SexpWrapper` (worker-side).

    On unpickle, attempts to resolve back to a cached `Sexp` via `get_sexp()`.
    """
    return (get_sexp, (obj._rid,))


def register_global_pickle_overrides() -> None:
    """Register global pickle reducers for `Sexp` and `SexpWrapper`."""
    copyreg.pickle(Sexp, _reduce_sexp)
    copyreg.pickle(SexpWrapper, _reduce_sexpwrapper)


# Make sure this runs before any pickling starts in the worker:
register_global_pickle_overrides()
