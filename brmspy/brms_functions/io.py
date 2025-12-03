import typing
import pandas as pd
import rpy2.robjects.packages as rpackages
from rpy2.robjects import default_converter, pandas2ri, numpy2ri, ListVector
from rpy2.robjects.conversion import localconverter

from ..helpers.singleton import _get_brms
from ..helpers.conversion import (
    brmsfit_to_idata,
    kwargs_r
)
from ..types import (
    FitResult, RListVectorExtension
)
import rpy2.robjects as ro


def get_brms_data(dataset_name: str) -> pd.DataFrame:
    """
    Load example dataset from brms package.
    
    Parameters
    ----------
    dataset_name : str
        Dataset name. Available datasets include:
        - 'epilepsy': Epileptic seizure counts
        - 'kidney': Kidney infection data with censoring
        - 'inhaler': Inhaler usage study
        - 'btdata': British Telecom share price data
        - And many more from brms package
    
    Returns
    -------
    pd.DataFrame
        Dataset as pandas DataFrame with column names preserved
    
    See Also
    --------
    brms::brmsdata : R documentation for available datasets
        https://paulbuerkner.com/brms/reference/index.html#data
    
    Examples
    --------
    Load epilepsy dataset:
    
    ```python
    from brmspy import brms
    epilepsy = brms.get_brms_data("epilepsy")
    print(epilepsy.head())
    print(epilepsy.columns)
    ```

    Load kidney dataset with censoring:
    
    ```python
    kidney = brms.get_brms_data("kidney")
    print(f"Shape: {kidney.shape}")
    print(f"Censored observations: {kidney['censored'].sum()}")
    ```
    """
    brms = _get_brms()
    with localconverter(default_converter + pandas2ri.converter + numpy2ri.converter) as cv:
        brmsdata = rpackages.data(brms)
        if brmsdata:
            return pd.DataFrame(brmsdata.fetch(dataset_name)[dataset_name])
        else:
            return pd.DataFrame({})




def save_rds(object: typing.Union[RListVectorExtension, ListVector], file: str, **kwargs) -> None:
    """
    Save brmsfit object or R object to RDS file.
    
    Saves fitted brms models or other R objects to disk using R's saveRDS() function.
    This allows persisting models for later use, sharing fitted models, or creating
    model checkpoints during long computations.
    
    Parameters
    ----------
    object : FitResult or ListVector
        Object to save. Can be:
        - FitResult from fit() - saves the underlying brmsfit R object
        - Any R ListVector object
    file : str
        File path where object will be saved. Typically uses .rds extension
        but not required
    **kwargs : dict
        Additional arguments passed to R's saveRDS():
        
        - compress : bool or str - Compression method:
            True (default), False, "gzip", "bzip2", "xz"
        - version : int - RDS format version (2 or 3)
        - ascii : bool - Use ASCII representation (default False)
        - refhook : function - Reference hook for serialization (NOT tested)
    
    Returns
    -------
    None
    
    See Also
    --------
    read_rds_fit : Load saved brmsfit as FitResult
    read_rds_raw : Load saved object as raw R ListVector
    fit : Fit models that can be saved
    
    Examples
    --------
    Save a fitted model:
    
    ```python
    from brmspy import brms
    
    # Fit model
    model = brms.fit(
        formula="y ~ x + (1|group)",
        data=data,
        family="gaussian",
        chains=4
    )
    
    # Save to file
    brms.save_rds(model, "my_model.rds")
    ```
    
    Save with compression options:
    
    ```python
    # High compression for storage
    brms.save_rds(model, "model.rds", compress="xz")
    
    # No compression for faster saving
    brms.save_rds(model, "model.rds", compress=False)
    ```
    
    Save and later reload:
    
    ```python
    # Save model
    brms.save_rds(model, "model.rds")
    
    # Later session: reload model
    loaded_model = brms.read_rds_fit("model.rds")
    
    # Use loaded model for predictions
    predictions = brms.posterior_predict(loaded_model, newdata=new_data)
    ```
    """
    if isinstance(object, RListVectorExtension):
        brmsfit = object.r
    else:
        brmsfit = object

    kwargs = kwargs_r(kwargs)
    
    r_save_rds = typing.cast(typing.Callable, ro.r('saveRDS'))
    r_save_rds(brmsfit, file, **kwargs)


def read_rds_raw(file: str, **kwargs) -> ListVector:
    """
    Load R object from RDS file as raw ListVector.
    
    Reads an RDS file and returns the raw R object without any Python conversion
    or processing. Useful when you need direct access to the R object structure
    or want to inspect saved objects before full conversion.
    
    Parameters
    ----------
    file : str
        Path to RDS file to load
    **kwargs : dict
        Additional arguments passed to R's readRDS():
        
        - refhook : function - Reference hook for deserialization
    
    Returns
    -------
    ListVector
        Raw R ListVector object from the RDS file
    
    See Also
    --------
    read_rds_fit : Load as FitResult with arviz InferenceData
    save_rds : Save R objects to RDS files
    
    Examples
    --------
    Load raw R object:
    
    ```python
    from brmspy import brms
    
    # Load raw brmsfit object
    raw_model = brms.read_rds_raw("model.rds")
    
    # Access R object directly (for advanced users)
    print(type(raw_model))  # rpy2.robjects.vectors.ListVector
    ```
    
    Inspect object structure before conversion:
    
    ```python
    # Load raw to check what's in the file
    raw_obj = brms.read_rds_raw("unknown_object.rds")
    
    # Inspect R object attributes
    print(raw_obj.names)
    
    # Then decide how to process it
    if "fit" in raw_obj.names:
        # It's a brmsfit, convert properly
        full_model = brms.read_rds_fit("unknown_object.rds")
    ```
    """
    r_read_rds = typing.cast(typing.Callable, ro.r('readRDS'))

    kwargs = kwargs_r(kwargs)
    brmsobject = r_read_rds(file, **kwargs)
    return brmsobject


def read_rds_fit(file: str, **kwargs) -> FitResult:
    """
    Load saved brmsfit object as FitResult with arviz InferenceData.
    
    Reads a brmsfit object from an RDS file and converts it to a FitResult
    with both arviz InferenceData (.idata) and the raw R object (.r).
    This is the recommended way to load saved brms models for analysis
    and predictions in Python.
    
    Parameters
    ----------
    file : str
        Path to RDS file containing saved brmsfit object
    **kwargs : dict
        Additional arguments passed to R's readRDS():
        
        - refhook : function - Reference hook for deserialization
    
    Returns
    -------
    FitResult
        Object with two attributes:
        - .idata : arviz.InferenceData with posterior samples and diagnostics
        - .r : R brmsfit object for use with brms functions
    
    Raises
    ------
    Exception
        If file doesn't exist or doesn't contain a valid brmsfit object
    
    See Also
    --------
    save_rds : Save brmsfit objects to RDS files
    read_rds_raw : Load as raw R object without conversion
    fit : Create brmsfit objects to save
    
    Examples
    --------
    Basic loading and analysis:
    
    ```python
    from brmspy import brms
    import arviz as az
    
    # Load previously saved model
    model = brms.read_rds_fit("my_model.rds")
    
    # Analyze with arviz
    az.summary(model.idata)
    az.plot_trace(model.idata)
    
    # Check diagnostics
    print(az.rhat(model.idata))
    ```
    
    Load and make predictions:
    
    ```python
    import pandas as pd
    
    # Load saved model
    model = brms.read_rds_fit("trained_model.rds")
    
    # Create new data for predictions
    newdata = pd.DataFrame({
        'x': [1.0, 2.0, 3.0],
        'group': ['A', 'B', 'A']
    })
    
    # Generate predictions
    predictions = brms.posterior_predict(model, newdata=newdata)
    print(predictions.idata.posterior_predictive)
    ```
    
    Load model for comparison:
    
    ```python
    # Load multiple saved models
    model1 = brms.read_rds_fit("model1.rds")
    model2 = brms.read_rds_fit("model2.rds")
    
    # Compare with arviz
    comparison = az.compare({
        'model1': model1.idata,
        'model2': model2.idata
    })
    print(comparison)
    ```
    
    Resume analysis from checkpoint:
    
    ```python
    # Load model from checkpoint during long computation
    try:
        model = brms.read_rds_fit("checkpoint.rds")
        print("Loaded from checkpoint")
    except:
        # Checkpoint doesn't exist, fit from scratch
        model = brms.fit(formula="y ~ x", data=data, chains=4)
        brms.save_rds(model, "checkpoint.rds")
    
    # Continue analysis
    summary = brms.summary(model)
    ```
    """
    brmsfit = read_rds_raw(file, **kwargs)
    idata = brmsfit_to_idata(brmsfit)

    return FitResult(idata=idata, r=brmsfit)