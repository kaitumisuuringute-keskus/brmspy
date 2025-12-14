"""
Type aliases for R↔Python conversion functions.

brmspy performs most conversion inside the worker process (where embedded R
lives). Converters may optionally use shared memory
([`ShmPool`][brmspy.types.shm.ShmPool]) to avoid extra copies when moving large
numeric data across the main↔worker boundary.

This module only defines type aliases used by converter registries and helper
code under [`brmspy.helpers._rpy2`][brmspy.helpers._rpy2].

Notes
-----
- The main process must not import `rpy2.robjects`. These aliases only refer to
  the lower-level `rpy2.rinterface_lib.sexp.Sexp` handle type.
"""

from collections.abc import Callable
from typing import Any, Union

import arviz as az
import numpy as np
import pandas as pd
import xarray as xr
from rpy2.rinterface_lib.sexp import Sexp

from brmspy.types.shm import ShmPool

__all__ = ["PyObject", "R2pyConverter", "Py2rConverter"]


PyObject = Union[
    dict,
    list,
    str,
    float,
    int,
    np.dtype,
    None,
    Any,  # keep for now
    pd.DataFrame,
    pd.Series,
    np.ndarray,
    az.InferenceData,
    xr.DataArray,
    xr.Dataset,
]
"""
Union of common Python-side objects produced by R→Python conversion.

This is intentionally broad: brmspy frequently returns standard scientific Python
types (NumPy/pandas/xarray/ArviZ), plus plain dict/list primitives.

Note
----
Avoid adding `Any` here unless absolutely necessary; it defeats the purpose of
having this alias.
"""


R2pyConverter = Callable[[Any, ShmPool | None], PyObject]
"""
Callable signature for an R→Python converter.

Parameters
----------
obj : Any
    R-side object (usually an rpy2 wrapper type).
shm : ShmPool or None
    Shared memory pool used to allocate backing buffers for large numeric payloads.

Returns
-------
PyObject
"""


Py2rConverter = Callable[[Any], Sexp]
"""
Callable signature for a Python→R converter.

Parameters
----------
obj : Any
    Python object to convert.

Returns
-------
rpy2.rinterface_lib.sexp.Sexp
    Low-level R object handle.
"""
