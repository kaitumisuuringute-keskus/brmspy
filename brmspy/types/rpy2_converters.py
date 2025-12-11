from typing import Any, Callable, Dict, Union, List
import pandas as pd
import numpy as np
import arviz as az
import xarray as xr
from rpy2.rinterface_lib.sexp import Sexp
from brmspy.types.shm import ShmPool
from dataclasses import dataclass

PyObject = Union[
    Dict,
    List,
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
