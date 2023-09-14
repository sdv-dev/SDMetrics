"""Single table quality report."""
from sdmetrics.reports._results_handler import QualityReportResultsHandler
from sdmetrics.reports.base_report import BaseReport
from sdmetrics.reports.single_table._properties import ColumnPairTrends, ColumnShapes


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
        self._results_handler = QualityReportResultsHandler()

    def _handle_results(self, verbose):
        self._results_handler.print_results(self._properties, self._overall_score, verbose)

    def get_score(self):
        """Return the overall quality score.

        Returns:
            float
                The overall quality score.
        """
        self._check_report_generated()
        return self._overall_score
