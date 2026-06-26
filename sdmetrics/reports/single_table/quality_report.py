"""Single table quality report."""

from sdmetrics.reports.base_report import BaseReport
from sdmetrics.reports.single_table._properties import ColumnPairTrends, ColumnShapes
from sdmetrics.reports.utils import (
    DEFAULT_REAL_ASSOCIATION_THRESHOLD,
    DEFAULT_REAL_CORRELATION_THRESHOLD,
    _warn_deprecated_report,
)


class QualityReport(BaseReport):
    """Single table quality report.

    This class creates a quality report for single-table data. It calculates the quality
    score along two properties - Column Shapes and Column Pair Trends.
    """

    def __init__(self):
        _warn_deprecated_report('single table quality', 'QualityReport')
        super().__init__()
        self.real_correlation_threshold = DEFAULT_REAL_CORRELATION_THRESHOLD
        self.real_association_threshold = DEFAULT_REAL_ASSOCIATION_THRESHOLD
        self._properties = {
            'Column Shapes': ColumnShapes(),
            'Column Pair Trends': ColumnPairTrends(),
        }
