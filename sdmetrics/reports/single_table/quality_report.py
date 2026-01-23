"""Single table quality report."""

from sdmetrics.reports.base_report import BaseReport
from sdmetrics.reports.single_table._properties import ColumnPairTrends, ColumnShapes


class QualityReport(BaseReport):
    """Single table quality report.

    This class creates a quality report for single-table data. It calculates the quality
    score along two properties - Column Shapes and Column Pair Trends.
    """

    def __init__(self):
        super().__init__()
        self.real_correlation_threshold = 0.5
        self.real_association_threshold = 0.3
        self._properties = {
            'Column Shapes': ColumnShapes(),
            'Column Pair Trends': ColumnPairTrends(),
        }
