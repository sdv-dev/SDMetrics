"""
This module implements univariate goodness-of-fit tests.
"""
from sdmetrics.multivariate.statistical.univariate.base import UnivariateMetric
from sdmetrics.multivariate.statistical.univariate.cstest import CSTest
from sdmetrics.multivariate.statistical.univariate.kstest import KSTest

__all__ = ["UnivariateMetric", "CSTest", "KSTest"]
