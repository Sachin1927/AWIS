"""Machine Learning module for AWIS"""

from .inference_attrition import get_predictor
from .inference_forecast import get_forecaster
from .inference_mobility import get_analyzer

__all__ = [
    'get_predictor',
    'get_forecaster',
    'get_analyzer'
]