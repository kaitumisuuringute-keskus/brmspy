import os

import os
from collections.abc import Mapping
from typing import Any

import numpy as np
import pandas as pd
from rpy2.rinterface import LangSexpVector, ListSexpVector
from brmspy.helpers._rpy2._converters._formula import _py2r_formula_part
from brmspy.types.formula_dsl import FormulaPart

from brmspy.helpers._rpy2._converters._arrays import (
    _py2r_dataframe,
    _py2r_numpy,
    _r2py_dataframe,
    _r2py_matrix,
)
from brmspy.helpers._rpy2._converters._generic import (
    _py2r_fallback,
    _py2r_mapping,
    _r2py_fallback,
    _r2py_language,
)
from brmspy.helpers._rpy2._converters._vectors import (
    _py2r_list,
    _r2py_listvector,
    _r2py_vector,
)
from brmspy.types.formula_dsl import FormulaPart

from ....types.rpy2_converters import Py2rConverter, R2pyConverter

from . import _registry

if os.environ.get("BRMSPY_WORKER") == "1":
    import rpy2.robjects as ro
    from rpy2.robjects.functions import SignatureTranslatedFunction

    _registry._R2PY_CONVERTERS.update(
        {
            ro.DataFrame: _r2py_dataframe,
            ro.Matrix: _r2py_matrix,
            ListSexpVector: _r2py_listvector,
            (ro.Formula, LangSexpVector, SignatureTranslatedFunction): _r2py_language,
            ro.vectors.Vector: _r2py_vector,  # must come AFTER specific vector types
            object: _r2py_fallback,
        }
    )
    _registry._PY2R_CONVERTERS.update(
        {
            FormulaPart: _py2r_formula_part,
            pd.DataFrame: _py2r_dataframe,
            np.ndarray: _py2r_numpy,
            Mapping: _py2r_mapping,
            # dont add Sequence, str is a sequence too!
            (list, tuple): _py2r_list,
            object: _py2r_fallback,
        }
    )


from ._dispatch import py_to_r, r_to_py

__all__ = ["py_to_r", "r_to_py"]
