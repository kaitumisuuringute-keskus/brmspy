"""
I/O helpers for brmspy.

This module contains helpers for:
- loading example datasets from R packages
- saving/loading R objects via ``saveRDS`` / ``readRDS``

Notes
-----
These functions are executed inside the worker process that hosts the embedded R
session.
"""

import typing

import pandas as pd
from rpy2.rinterface import ListSexpVector

from ..helpers._rpy2._conversion import brmsfit_to_idata, kwargs_r, r_to_py
from ..types.brms_results import FitResult, ProxyListSexpVector, RListVectorExtension


def get_data(dataset_name: str, **kwargs) -> pd.DataFrame:
    """
    Load an R dataset and return it as a pandas DataFrame.

    This is a thin wrapper around R's ``data()`` that loads the object
    into the R global environment and converts it to a
    :class:`pandas.DataFrame`.

    Parameters
    ----------
    dataset_name : str
        Name of the dataset as used in R (e.g. ``"BTdata"``).
    **kwargs
        Additional keyword arguments forwarded to R's ``data()`` function,
        for example ``package="MCMCglmm"`` or other arguments supported
        by ``utils::data()`` in R.

    Returns
    -------
    pd.DataFrame
        Dataset converted to a pandas DataFrame.

    Raises
    ------
    KeyError
        If the dataset is not found in the R global environment after
        calling ``data()``.
    RuntimeError
        If conversion from the R object to a pandas DataFrame fails.

    See Also
    --------
    get_brms_data
        Convenience wrapper for datasets from the ``brms`` package.
    """
    import rpy2.robjects as ro

    r_kwargs = kwargs_r(kwargs)

    r_data = typing.cast(typing.Callable, ro.r["data"])
    r_data(dataset_name, **r_kwargs)
    r_obj = ro.globalenv[dataset_name]

    return typing.cast(pd.DataFrame, r_to_py(r_obj))


def get_brms_data(dataset_name: str, **kwargs) -> pd.DataFrame:
    """
    Load an example dataset from the R ``brms`` package.

    Parameters
    ----------
    dataset_name : str
        Dataset name (for example ``"epilepsy"`` or ``"kidney"``).
    **kwargs
        Forwarded to R ``utils::data()`` via `get_data()`.

    Returns
    -------
    pandas.DataFrame
        Dataset converted to a DataFrame.

    Examples
    --------
    ```python
    from brmspy import brms

    epilepsy = brms.get_brms_data("epilepsy")
    assert epilepsy.shape[0] > 0
    ```
    """
    return get_data(dataset_name, package="brms", **kwargs)


def save_rds(
    object: RListVectorExtension | ProxyListSexpVector, file: str, **kwargs
) -> None:
    """
    Save an R object to an ``.rds`` file via R ``saveRDS()``.

    Parameters
    ----------
    object : RListVectorExtension or ProxyListSexpVector
        Object to save. If you pass a `FitResult`, the underlying brmsfit is saved.
    file : str
        Output path.
    **kwargs
        Forwarded to R ``saveRDS()`` (for example ``compress="xz"``).

    Returns
    -------
    None

    Examples
    --------
    ```python
    from brmspy import brms

    model = brms.brm("y ~ x", data=df, chains=4)
    brms.save_rds(model, "model.rds")
    ```
    """
    import rpy2.robjects as ro

    if isinstance(object, RListVectorExtension):
        brmsfit = object.r
    else:
        brmsfit = object

    kwargs = kwargs_r(kwargs)

    r_save_rds = typing.cast(typing.Callable, ro.r("saveRDS"))
    r_save_rds(brmsfit, file, **kwargs)


def read_rds_raw(file: str, **kwargs) -> ListSexpVector:
    """
    Load an R object from an ``.rds`` file via R ``readRDS()``.

    This returns the raw R object handle.

    Parameters
    ----------
    file : str
        Input path.
    **kwargs
        Forwarded to R ``readRDS()``.

    Returns
    -------
    rpy2.rinterface.ListSexpVector
        Raw R object.

    Examples
    --------
    ```python
    from brmspy import brms

    obj = brms.read_rds_raw("model.rds")
    ```
    """
    import rpy2.robjects as ro

    r_read_rds = typing.cast(typing.Callable, ro.r("readRDS"))

    kwargs = kwargs_r(kwargs)
    brmsobject = r_read_rds(file, **kwargs)
    return brmsobject


def read_rds_fit(file: str, **kwargs) -> FitResult:
    """
    Load a saved brms model from an ``.rds`` file.

    Parameters
    ----------
    file : str
        Input path containing a saved brmsfit.
    **kwargs
        Forwarded to R ``readRDS()``.

    Returns
    -------
    FitResult
        `FitResult` containing ArviZ `InferenceData` and an underlying R handle.

    Examples
    --------
    ```python
    from brmspy import brms

    fit = brms.read_rds_fit("model.rds")
    fit.idata.posterior
    ```
    """
    brmsfit = read_rds_raw(file, **kwargs)
    idata = brmsfit_to_idata(brmsfit)

    return FitResult(idata=idata, r=brmsfit)
