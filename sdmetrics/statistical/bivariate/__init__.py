"""
This module implements bivariate KL-divergence/relative-entropy measures.
"""

from sdmetrics.statistical.bivariate.base import BivariateMetric
from sdmetrics.statistical.bivariate.continuous import ContinuousDivergence
from sdmetrics.statistical.bivariate.discrete import DiscreteDivergence

__all__ = ["BivariateMetric", "DiscreteDivergence", "ContinuousDivergence"]
