"""Single table properties for sdmetrics."""

from sdmetrics.reports.single_table._properties.base import BaseSingleTableProperty
from sdmetrics.reports.single_table._properties.boundary import Boundary
from sdmetrics.reports.single_table._properties.column_pair_trends import ColumnPairTrends
from sdmetrics.reports.single_table._properties.column_shapes import ColumnShapes
from sdmetrics.reports.single_table._properties.coverage import Coverage
from sdmetrics.reports.single_table._properties.data_validity import DataValidity
from sdmetrics.reports.single_table._properties.structure import Structure
from sdmetrics.reports.single_table._properties.synthesis import Synthesis

__all__ = [
    'BaseSingleTableProperty',
    'ColumnShapes',
    'ColumnPairTrends',
    'Coverage',
    'Boundary',
    'Synthesis',
    'Structure',
    'DataValidity',
]
