"""
ArviZ type definitions — version-aware base classes.

This module lives in ``types/`` (not ``helpers/``) so that other ``types``
modules can import from it without creating an import cycle.

``helpers.arviz_compat`` re-exports these and adds runtime helpers.
"""

from __future__ import annotations

import arviz as az
import xarray as xr
from packaging.version import Version

__all__ = [
    "ARVIZ_V1",
    "InferenceDataBase",
    "InferenceDataLike",
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
