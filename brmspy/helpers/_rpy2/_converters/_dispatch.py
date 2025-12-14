from rpy2.rinterface_lib.sexp import Sexp

from brmspy.types.brms_results import RListVectorExtension
from brmspy.types.shm import ShmPool

from ....types.rpy2_converters import PyObject
from . import _registry


def r_to_py(obj: Sexp, shm: ShmPool | None = None) -> PyObject:
    """
    Convert R objects to Python objects via rpy2.

    Comprehensive converter that handles R lists (named/unnamed), vectors,
    formulas, and language objects. Provides sensible Python equivalents
    for all R types with special handling for edge cases.

    Parameters
    ----------
    obj : rpy2 R object
        R object to convert to Python

    Returns
    -------
    any
        Python representation of the R object:
        - R NULL → None
        - Named list → dict (recursively)
        - Unnamed list → list (recursively)
        - Length-1 vector → scalar (int, float, str, bool)
        - Length-N vector → list of scalars
        - Formula/Language object → str (descriptive representation)
        - Other objects → default rpy2 conversion or str fallback

    Notes
    -----
    **Conversion Rules:**

    1. **R NULL**: → Python None
    2. **Atomic vectors** (numeric, character, logical):
       - Length 1: → Python scalar (int, float, str, bool)
       - Length >1: → Python list of scalars
    3. **Named lists** (ListVector with names): → Python dict, recursively
    4. **Unnamed lists**: → Python list, recursively
    5. **Formulas** (e.g., `y ~ x`): → String representation
    6. **Language objects** (calls, expressions): → String representation
    7. **Functions**: → String representation
    8. **Everything else**: Try default rpy2 conversion, fallback to string

    **Recursive Conversion:**

    List elements and dictionary values are recursively converted:
    ```R
    list(a = list(b = c(1, 2)))  →  {'a': {'b': [1, 2]}}
    ```

    **Safe Fallback:**

    R language objects, formulas, and functions are converted to descriptive
    strings rather than attempting complex conversions that might fail.

    Examples
    --------

    ```python
    from brmspy.helpers.conversion import r_to_py
    import rpy2.robjects as ro

    # R NULL
    r_to_py(ro.NULL)  # None

    # Scalars
    r_to_py(ro.IntVector([5]))    # 5
    r_to_py(ro.FloatVector([3.14]))  # 3.14
    r_to_py(ro.StrVector(["hello"]))  # "hello"

    # Vectors
    r_to_py(ro.IntVector([1, 2, 3]))  # [1, 2, 3]
    ```

    See Also
    --------
    py_to_r : Convert Python objects to R
    brmspy.brms.summary : Returns Python-friendly summary dict
    """
    import rpy2.robjects as ro

    from brmspy._singleton._shm_singleton import _get_shm

    if obj is ro.NULL:
        return None

    _type = type(obj)
    converter = None

    if shm is None:
        shm = _get_shm()

    if _type in _registry._R2PY_CONVERTERS:
        # O(1) lookup first
        converter = _registry._R2PY_CONVERTERS[_type]
    else:
        for _type, _con in _registry._R2PY_CONVERTERS.items():
            if isinstance(obj, _type):
                converter = _con
                break

    assert len(_registry._R2PY_CONVERTERS) > 0, "NO R2PY CONVERTERS"
    assert (
        converter
    ), "object fallback must be in place in __init__.py! This is an issue with the library, not the user!"
    return converter(obj, shm)


def py_to_r(obj: PyObject) -> Sexp:
    """
    Convert arbitrary Python objects to R objects via rpy2.

    Comprehensive converter that handles nested structures (dicts, lists),
    DataFrames, arrays, and scalars. Uses rpy2's converters with special
    handling for dictionaries (→ R named lists) and lists of dicts.

    Parameters
    ----------
    obj : any
        Python object to convert. Supported types:
        - None → R NULL
        - dict → R named list (ListVector), recursively
        - list/tuple of dicts → R list of named lists
        - list/tuple (other) → R vector or list
        - pd.DataFrame → R data.frame
        - np.ndarray → R vector/matrix
        - scalars (int, float, str, bool) → R atomic types

    Returns
    -------
    rpy2 R object
        R representation of the Python object

    Notes
    -----
    **Conversion Rules:**

    1. **None**: → R NULL
    2. **DataFrames**: → R data.frame (via pandas2ri)
    3. **Dictionaries**: → R named list (ListVector), recursively converting values
    4. **Lists of dicts**: → R list with 1-based indexed names containing named lists
    5. **Other lists/tuples**: → R vectors or lists (via rpy2 default)
    6. **NumPy arrays**: → R vectors/matrices (via numpy2ri)
    7. **Scalars**: → R atomic values

    **Recursive Conversion:**

    Dictionary values are recursively converted, allowing nested structures:
    ```python
    {'a': {'b': [1, 2, 3]}}  →  list(a = list(b = c(1, 2, 3)))
    ```

    **List of Dicts:**

    Lists containing only dicts are converted to R lists with 1-based indexing:
    ```python
    [{'x': 1}, {'x': 2}]  →  list("1" = list(x = 1), "2" = list(x = 2))
    ```

    Examples
    --------

    ```python
    from brmspy.helpers.conversion import py_to_r
    import numpy as np
    import pandas as pd

    # Scalars
    py_to_r(5)        # R: 5
    py_to_r("hello")  # R: "hello"
    py_to_r(None)     # R: NULL

    # Arrays
    py_to_r(np.array([1, 2, 3]))  # R: c(1, 2, 3)

    # DataFrames
    df = pd.DataFrame({'x': [1, 2], 'y': [3, 4]})
    py_to_r(df)  # R: data.frame(x = c(1, 2), y = c(3, 4))
    ```

    See Also
    --------
    r_to_py : Convert R objects back to Python
    kwargs_r : Convert keyword arguments dict for R function calls
    brmspy.brms.fit : Uses this for converting data to R
    """
    import rpy2.robjects as ro

    if obj is None:
        return ro.NULL

    if isinstance(obj, ro.Sexp):
        return obj

    if isinstance(obj, RListVectorExtension) and isinstance(obj.r, ro.Sexp):
        return obj.r

    _type = type(obj)
    converter = None

    if _type in _registry._PY2R_CONVERTERS:
        # O(1) lookup first
        converter = _registry._PY2R_CONVERTERS[_type]
    else:
        for _type, _con in _registry._PY2R_CONVERTERS.items():
            if isinstance(obj, _type):
                converter = _con
                break

    assert len(_registry._PY2R_CONVERTERS) > 0, "NO PY2R CONVERTERS"
    assert (
        converter
    ), "object fallback must be in place in __init__.py! This is an issue with the library, not the user!"
    return converter(obj)
