"""Multi table properties for sdmetrics."""

from sdmetrics.reports.multi_table._properties.base import BaseMultiTableProperty
from sdmetrics.reports.multi_table._properties.cardinality import Cardinality

__all__ = [
    'BaseMultiTableProperty',
    'Cardinality',
    'ColumnShapes',
    'ColumnPairTrends'
]
