from dataclasses import dataclass, is_dataclass
from types import UnionType
from typing import Any, Callable, Dict, List, Optional, Union, cast, get_args, get_origin
import pandas as pd
import rpy2.robjects as ro

from brmspy.helpers.log import log_warning


@dataclass
class IterConf:
    cls: type[Any] = dict
    optional: bool = False


_UNION_TYPES = (Union, UnionType)


def _normalize_annotation(ann: Any) -> IterConf:
    """
    Turn a typing annotation (Dict[str, int], Optional[dict], pd.DataFrame, ...)
    into (runtime_cls, optional_flag).
    """
    origin = get_origin(ann)

    # Handle Optional[T] / T | None
    if origin in _UNION_TYPES:
        args = get_args(ann)
        non_none = [a for a in args if a is not type(None)]

        # Exactly Union[T, NoneType] -> Optional[T]
        if len(non_none) == 1 and len(args) == 2:
            inner = non_none[0]
            inner_origin = get_origin(inner)
            if inner_origin is not None:
                return IterConf(cls=inner_origin, optional=True)
            elif isinstance(inner, type):
                return IterConf(cls=inner, optional=True)
            else:
                return IterConf(cls=object, optional=True)

        # Other weird unions: don't enforce at runtime
        return IterConf(cls=object, optional=False)

    # Normal generics: Dict[str, int] -> dict, list[int] -> list, etc.
    if origin is not None:
        return IterConf(cls=origin, optional=False)

    # Plain classes: str, int, pd.DataFrame, ...
    if isinstance(ann, type):
        return IterConf(cls=ann, optional=False)

    # Fallback
    return IterConf(cls=object, optional=False)


def _build_iterconf_from_dataclass(dc: type[Any]) -> dict[str, IterConf]:
    if not is_dataclass(dc):
        raise TypeError("target_dataclass must be a dataclass type")

    annotations = getattr(dc, "__annotations__", {})
    return {
        name: _normalize_annotation(tp)
        for name, tp in annotations.items()
    }


def _matches_iterconf(value: Any, conf: IterConf) -> bool:
    if value is None:
        return conf.optional
    return isinstance(value, conf.cls)


_fun_repr = cast(Callable, ro.r('function(x) paste(capture.output(x), collapse = "\\n")'))
def iterate_robject_to_dataclass(
    names: List[str],
    get: Callable[[str], Any],
    target_dataclass: type[Any],
    r: Optional[ro.ListVector],
    iteration_params: Optional[Dict[str, IterConf]] = None
):
    """
    Generic helper to iterate over R summary-like objects and convert them
    into a Python dataclass instance.

    - `names` is e.g. summary_r.names
    - `get(param)` should return the R slot already converted via rpy2 (or raw)
    - `target_dataclass` is a @dataclass whose field names mirror the params
    """
    if target_dataclass is None:
        raise ValueError("target_dataclass must be specified")

    if not is_dataclass(target_dataclass):
        raise TypeError("target_dataclass must be a dataclass type")

    if iteration_params is None:
        iteration_params = _build_iterconf_from_dataclass(target_dataclass)

    out: Dict[str, Any] = {}

    for param, iterconf in iteration_params.items():
        if param == "_str" and r:
            try:
                repr = cast(List[str], _fun_repr(r))
                repr = str(repr[0])
            except:
                repr = None
            out['_str'] = repr
            continue

        if param not in names:
            continue

        try:
            data = get(param)

            if data is None:
                if not iterconf.optional:
                    log_warning(
                        f"Param '{param}' is None, but IterConf does not mark it as optional. Skipping."
                    )
                    continue
            else:
                # Special handling for DataFrame fields
                if iterconf.cls is pd.DataFrame and not isinstance(data, pd.DataFrame):
                    try:
                        data = pd.DataFrame(data)
                    except Exception as e:
                        log_warning(
                            f"Failed to convert '{param}' object to pd.DataFrame. "
                            f"Passing default. {e}"
                        )

                if not _matches_iterconf(data, iterconf):
                    log_warning(
                        f"Type of param '{param}' {type(data)} does not match "
                        f"expected '{iterconf.cls}'"
                    )

            out[param] = data

        except Exception as e:
            log_warning(f"Failed to parse {param} into python object: {e}")

    if r:
        out['r'] = r

    # Build the dataclass instance
    return target_dataclass(**out)
