from typing import Any, Callable, Dict, Union, List
import pandas as pd
import numpy as np
import arviz as az
from rpy2.rinterface_lib.sexp import Sexp
from brmspy.session.transport import ShmPool

PyObject = Union[
    Dict, List, str, float, int, None, pd.DataFrame, np.ndarray, az.InferenceData
]

R2pyConverter = Callable[[Any, ShmPool | None], PyObject]
Py2rConverter = Callable[[Any], Sexp]
