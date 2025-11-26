from dataclasses import dataclass
from typing import Any
import rpy2.robjects as robjects

@dataclass
class FitResult:
    idata: 'arviz.InferenceData'
    r: robjects.ListVector

@dataclass
class PosteriorEpredResult:
    idata: 'arviz.InferenceData'
    r: robjects.ListVector

@dataclass
class PosteriorPredictResult:
    idata: 'arviz.InferenceData'
    r: robjects.ListVector