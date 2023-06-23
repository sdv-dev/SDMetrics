"""Multi table properties for sdmetrics."""

from sdmetrics.reports.multi_table._properties.base import BaseMultiTableProperty
from sdmetrics.reports.multi_table._properties.relationship import (
    CardinalityShapeSimilarityProperty)

__all__ = [
    'BaseMultiTableProperty',
    'CardinalityShapeSimilarityProperty'
]
