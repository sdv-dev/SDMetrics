"""Multi table properties for sdmetrics."""

from sdmetrics.reports.multi_table._properties.base import BaseMultiTableProperty
from sdmetrics.reports.multi_table._properties.cardinality import Cardinality
from sdmetrics.reports.multi_table._properties.column_pair_trends import ColumnPairTrends
from sdmetrics.reports.multi_table._properties.column_shapes import ColumnShapes
from sdmetrics.reports.multi_table._properties.coverage import Coverage

__all__ = [
    'BaseMultiTableProperty',
    'Cardinality',
    'ColumnShapes',
    'ColumnPairTrends',
    'Coverage'
]
