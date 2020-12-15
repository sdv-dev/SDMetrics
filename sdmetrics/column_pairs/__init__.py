"""Metrics to compare column pairs."""

from sdmetrics.column_pairs import statistical
from sdmetrics.column_pairs.base import ColumnPairsMetric
from sdmetrics.column_pairs.statistical.kl_divergence import (
    ContinuousKLDivergence, DiscreteKLDivergence)

__all__ = [
    'statistical',
    'ColumnPairsMetric',
    'ContinuousKLDivergence',
    'DiscreteKLDivergence',
]
