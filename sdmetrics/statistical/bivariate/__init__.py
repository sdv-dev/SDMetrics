"""
This module implements bivariate KL-divergence/relative-entropy measures.
"""
from .base import BivariateMetric
from .continuous import ContinuousDivergence
from .discrete import DiscreteDivergence

__all__ = ["BivariateMetric", "DiscreteDivergence", "ContinuousDivergence"]
