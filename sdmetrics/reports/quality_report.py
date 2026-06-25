"""Unified quality report."""

from sdmetrics.reports.base_unified_report import BaseUnifiedReport
from sdmetrics.reports.multi_table._properties import (
    Cardinality,
    ColumnPairTrends,
    ColumnShapes,
    InterTableTrends,
)
from sdmetrics.reports.utils import (
    DEFAULT_REAL_ASSOCIATION_THRESHOLD,
    DEFAULT_REAL_CORRELATION_THRESHOLD,
)


class QualityReport(BaseUnifiedReport):
    """Quality report for single-table and multi-table data."""

    def __init__(self):
        super().__init__()
        self.real_correlation_threshold = DEFAULT_REAL_CORRELATION_THRESHOLD
        self.real_association_threshold = DEFAULT_REAL_ASSOCIATION_THRESHOLD
        self._properties = {
            'Column Shapes': ColumnShapes(),
            'Column Pair Trends': ColumnPairTrends(),
            'Cardinality': Cardinality(),
            'Intertable Trends': InterTableTrends(),
        }
