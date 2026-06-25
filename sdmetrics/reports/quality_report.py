"""Unified quality report."""

from sdmetrics.reports.base_unified_report import BaseUnifiedReport
from sdmetrics.reports.multi_table._properties import (
    Cardinality,
    ColumnPairTrends,
    ColumnShapes,
    InterTableTrends,
)


class QualityReport(BaseUnifiedReport):
    """Quality report for single-table and multi-table data."""

    def __init__(self):
        super().__init__()
        self.real_correlation_threshold = 0.5
        self.real_association_threshold = 0.3
        self._properties = {
            'Column Shapes': ColumnShapes(),
            'Column Pair Trends': ColumnPairTrends(),
            'Cardinality': Cardinality(),
            'Intertable Trends': InterTableTrends(),
        }
