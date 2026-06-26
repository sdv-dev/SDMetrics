"""Multi table quality report."""

from sdmetrics.reports.multi_table._properties import (
    Cardinality,
    ColumnPairTrends,
    ColumnShapes,
    InterTableTrends,
)
from sdmetrics.reports.multi_table.base_multi_table_report import BaseMultiTableReport
from sdmetrics.reports.utils import (
    DEFAULT_REAL_ASSOCIATION_THRESHOLD,
    DEFAULT_REAL_CORRELATION_THRESHOLD,
    _warn_deprecated_report,
)


class QualityReport(BaseMultiTableReport):
    """Multi table quality report.

    This class creates a quality report for multi-table data. It calculates the quality
    score along three properties - Column Shapes, Column Pair Trends, and Cardinality.
    """

    def __init__(self):
        _warn_deprecated_report('multi table quality', 'QualityReport')
        super().__init__()
        self.real_correlation_threshold = DEFAULT_REAL_CORRELATION_THRESHOLD
        self.real_association_threshold = DEFAULT_REAL_ASSOCIATION_THRESHOLD
        self._properties = {
            'Column Shapes': ColumnShapes(),
            'Column Pair Trends': ColumnPairTrends(),
            'Cardinality': Cardinality(),
            'Intertable Trends': InterTableTrends(),
        }
