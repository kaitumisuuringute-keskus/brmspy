"""
ArviZ version compatibility layer.

Abstracts all version-dependent ArviZ behaviour so the rest of brmspy
never imports ``arviz`` directly.  Detects the installed version once at
import time and exposes a unified API that works identically on:

* ArviZ < 1.0  →  ``az.InferenceData``
* ArviZ ≥ 1.0  →  ``xarray.DataTree``

To drop < 1.0 support in the future, delete the ``else`` branches and simplify.
"""

from __future__ import annotations

from typing import Any, Union

import arviz as az
import xarray as xr
from packaging.version import Version

__all__ = [
    "ARVIZ_V1",
    "InferenceDataLike",
    "InferenceDataBase",
    "from_dict",
    "get_groups",
    "extend",
    "get_group_dataset",
    "set_group_dataset",
    "construct_from_datasets",
    "is_inference_data",
]

# ---------------------------------------------------------------------------
# Version detection (evaluated once at import time)
# ---------------------------------------------------------------------------

ARVIZ_V1: bool = Version(az.__version__) >= Version("1.0.0")
"""True when the installed ArviZ is ≥ 1.0 (DataTree-based)."""

# ---------------------------------------------------------------------------
# Canonical types
# ---------------------------------------------------------------------------

if ARVIZ_V1:
    InferenceDataLike = xr.DataTree
    """Runtime type returned by ``from_dict`` and stored on result objects."""

    InferenceDataBase: type = xr.DataTree
    """Base class for the typed InferenceData subclasses in ``brms_results``."""
else:
    InferenceDataLike = az.InferenceData  # type: ignore[misc]
    InferenceDataBase: type = az.InferenceData  # type: ignore[misc]


# ---------------------------------------------------------------------------
# Unified helpers
# ---------------------------------------------------------------------------


def from_dict(**kwargs: Any) -> InferenceDataLike:  # type: ignore[return]
    """
    Build an inference-data object from arrays/dicts.

    Wraps ``az.from_dict`` — returns ``InferenceData`` on < 1.0 and
    ``DataTree`` on ≥ 1.0.
    """
    return az.from_dict(**kwargs)


def get_groups(obj: InferenceDataLike) -> list[str]:
    """Return plain group names (no ``/`` prefix)."""
    if ARVIZ_V1:
        return list(obj.children)  # type: ignore[union-attr]
    else:
        return list(obj.groups())  # type: ignore[union-attr]


def extend(target: InferenceDataLike, source: InferenceDataLike) -> None:
    """Merge *source* groups into *target* **in-place**."""
    if ARVIZ_V1:
        target.update(source)  # type: ignore[union-attr]
    else:
        target.extend(source)  # type: ignore[union-attr]


def get_group_dataset(obj: InferenceDataLike, name: str) -> xr.Dataset:
    """
    Get a group as an ``xr.Dataset``.

    On < 1.0 this is ``getattr(obj, name)``; on ≥ 1.0 it is
    ``obj[name].dataset`` (which returns a ``DatasetView``, a Dataset subclass).
    """
    if ARVIZ_V1:
        return obj[name].dataset  # type: ignore[union-attr,index]
    else:
        return getattr(obj, name)


def set_group_dataset(
    obj: InferenceDataLike, name: str, ds: xr.Dataset
) -> None:
    """Attach *ds* as the group *name* on *obj*."""
    if ARVIZ_V1:
        obj[name] = xr.DataTree(dataset=ds)  # type: ignore[index]
    else:
        setattr(obj, name, ds)


def has_group(obj: InferenceDataLike, name: str) -> bool:
    """Check whether *obj* contains a group called *name*."""
    return name in get_groups(obj)


def construct_from_datasets(
    groups: dict[str, xr.Dataset],
    **kwargs: Any,
) -> InferenceDataLike:  # type: ignore[return]
    """
    Build an inference-data object from a ``{group_name: Dataset}`` dict.

    On < 1.0 this calls ``az.InferenceData(**groups, ...)``.
    On ≥ 1.0 this calls ``xr.DataTree.from_dict({"/name": ds, ...})``.
    """
    if ARVIZ_V1:
        tree_dict: dict[str, xr.Dataset] = {
            f"/{k}": v for k, v in groups.items()
        }
        return xr.DataTree.from_dict(tree_dict)
    else:
        return az.InferenceData(**groups, **kwargs)


def is_inference_data(obj: Any) -> bool:
    """Version-agnostic ``isinstance`` check."""
    if ARVIZ_V1:
        return isinstance(obj, xr.DataTree)
    else:
        return isinstance(obj, az.InferenceData)
