"""
This module implements univariate goodness-of-fit tests.
"""
from sdmetrics.statistical.univariate.base import UnivariateMetric
from sdmetrics.statistical.univariate.cstest import CSTest
from sdmetrics.statistical.univariate.kstest import KSTest

__all__ = ["UnivariateMetric", "CSTest", "KSTest"]
