"""
This module implements bivariate KL-divergence/relative-entropy measures.
"""

from sdmetrics.multivariate.statistical.bivariate.base import BivariateMetric
from sdmetrics.multivariate.statistical.bivariate.continuous import ContinuousDivergence
from sdmetrics.multivariate.statistical.bivariate.discrete import DiscreteDivergence

__all__ = ["BivariateMetric", "DiscreteDivergence", "ContinuousDivergence"]
