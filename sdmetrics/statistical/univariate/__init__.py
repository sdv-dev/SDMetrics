"""
This module implements univariate goodness-of-fit tests.
"""
from .base import UnivariateMetric
from .cstest import CSTest
from .kstest import KSTest

__all__ = ["UnivariateMetric", "CSTest", "KSTest"]
