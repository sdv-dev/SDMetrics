"""Single table properties for sdmetrics."""

from sdmetrics.reports.single_table._properties.base import BaseSingleTableProperty
from sdmetrics.reports.single_table._properties.column_shapes import ColumnShapes

__all__ = [
    'BaseSingleTableProperty',
    'ColumnShapes',
]