"""Multi table properties for sdmetrics."""

from sdmetrics.reports.multi_table._properties.base import BaseMultiTableProperty
from sdmetrics.reports.multi_table._properties.boundary import Boundary
from sdmetrics.reports.multi_table._properties.cardinality import Cardinality
from sdmetrics.reports.multi_table._properties.column_pair_trends import ColumnPairTrends
from sdmetrics.reports.multi_table._properties.column_shapes import ColumnShapes
from sdmetrics.reports.multi_table._properties.coverage import Coverage
from sdmetrics.reports.multi_table._properties.inter_table_trends import InterTableTrends
from sdmetrics.reports.multi_table._properties.synthesis import Synthesis

__all__ = [
    'BaseMultiTableProperty',
    'Boundary',
    'Cardinality',
    'ColumnShapes',
    'ColumnPairTrends',
    'Coverage',
    'InterTableTrends',
    'Synthesis',
]
