"""Result types for brmspy functions."""

from dataclasses import dataclass
from typing import Any, Dict
import rpy2.robjects as robjects

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
    idata: 'arviz.InferenceData'
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
    idata: 'arviz.InferenceData'
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
    idata: 'arviz.InferenceData'
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
    idata: 'arviz.InferenceData'
    r: robjects.ListVector

@dataclass
class FormulaResult:
    r: robjects.ListVector
    dict: Dict