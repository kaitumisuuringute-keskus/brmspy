from collections.abc import Callable
from typing import Any, Union

import arviz as az
import numpy as np
import pandas as pd
import xarray as xr
from rpy2.rinterface_lib.sexp import Sexp

from brmspy.types.shm import ShmPool

PyObject = Union[
    dict,
    list,
    str,
    float,
    int,
    np.dtype,
    None,
    Any,  # remove
    pd.DataFrame,
    pd.Series,
    np.ndarray,
    az.InferenceData,
    xr.DataArray,
    xr.Dataset,
]

R2pyConverter = Callable[[Any, ShmPool | None], PyObject]
Py2rConverter = Callable[[Any], Sexp]
