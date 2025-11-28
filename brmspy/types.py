"""Result types for brmspy functions."""

from dataclasses import dataclass
from typing import Dict
import rpy2.robjects as robjects
import arviz as az
import xarray as xr

# -----------------------------------------------------
# az.InferenceData extensions for proper typing in IDEs
# -----------------------------------------------------

class IDFit(az.InferenceData):
    posterior: xr.Dataset
    posterior_predictive: xr.Dataset
    log_likelihood: xr.Dataset
    observed_data: xr.Dataset
    coords: xr.Dataset
    dims: xr.Dataset

class IDEpred(az.InferenceData):
    posterior: xr.Dataset

class IDPredict(az.InferenceData):
    posterior_predictive: xr.Dataset

class IDLinpred(az.InferenceData):
    predictions: xr.Dataset

class IDLogLik(az.InferenceData):
    log_likelihood: xr.Dataset




# ---------------------
# Function return types
# ---------------------

@dataclass
class GenericResult:
    """Generic result container with arviz and R objects.
    
    Attributes
    ----------
    idata : arviz.InferenceData
        arviz InferenceData object
    r : robjects.ListVector
        R object from brms
    """
    idata: az.InferenceData
    r: robjects.ListVector

@dataclass
class FitResult:
    """Result from fit() function.
    
    Attributes
    ----------
    idata : arviz.InferenceData
        arviz InferenceData with posterior, posterior_predictive,
        log_likelihood, and observed_data groups
    r : robjects.ListVector
        brmsfit R object from brms::brm()
    """
    idata: IDFit
    r: robjects.ListVector

@dataclass
class PosteriorEpredResult:
    """Result from posterior_epred() function.
    
    Attributes
    ----------
    idata : arviz.InferenceData
        arviz InferenceData with expected values in 'posterior' group
    r : robjects.ListVector
        R matrix from brms::posterior_epred()
    """
    idata: IDEpred
    r: robjects.ListVector

@dataclass
class PosteriorPredictResult:
    """Result from posterior_predict() function.
    
    Attributes
    ----------
    idata : arviz.InferenceData
        arviz InferenceData with predictions in 'posterior_predictive' group
    r : robjects.ListVector
        R matrix from brms::posterior_predict()
    """
    idata: IDPredict
    r: robjects.ListVector

@dataclass
class LogLikResult:
    idata: IDLogLik
    r: robjects.ListVector

@dataclass
class PosteriorLinpredResult:
    idata: IDLinpred
    r: robjects.ListVector

@dataclass
class FormulaResult:
    r: robjects.ListVector
    dict: Dict