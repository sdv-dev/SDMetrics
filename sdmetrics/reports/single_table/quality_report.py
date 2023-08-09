"""Single table quality report."""
from sdmetrics.reports.base_report import BaseReport
from sdmetrics.reports.single_table._properties import ColumnPairTrends, ColumnShapes
from sdmetrics.reports.utils import _print_results_quality_report


class QualityReport(BaseReport):
    """Single table quality report.

    This class creates a quality report for single-table data. It calculates the quality
    score along two properties - Column Shapes and Column Pair Trends.
    """

    def __init__(self):
        super().__init__()
        self._properties = {
            'Column Shapes': ColumnShapes(),
            'Column Pair Trends': ColumnPairTrends()
        }

    def _print_results(self):
        """Print the results of the report."""
        _print_results_quality_report(self)

    def get_score(self):
        """Return the overall quality score.

        Returns:
            float
                The overall quality score.
        """
        self._check_report_generated()
        return self._overall_score
