"""Univariate goodness-of-fit tests."""

from sdmetrics.single_column.statistical.cstest import CSTest
from sdmetrics.single_column.statistical.kscomplement import KSComplement
from sdmetrics.single_column.statistical.statistic_similarity import StatisticSimilarity

__all__ = [
    'CSTest',
    'KSComplement',
    'StatisticSimilarity'
]
