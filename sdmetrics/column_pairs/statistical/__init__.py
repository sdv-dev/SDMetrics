"""Statistical Metrics to compare column pairs."""

from sdmetrics.column_pairs.statistical.kl_divergence import (
    ContinuousKLDivergence, DiscreteKLDivergence)

__all__ = [
    'ContinuousKLDivergence',
    'DiscreteKLDivergence',
]
