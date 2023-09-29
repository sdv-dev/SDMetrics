"""Multi table quality report."""
from sdmetrics.reports._results_handler import QualityReportResultsHandler
from sdmetrics.reports.multi_table._properties import (
    Cardinality, ColumnPairTrends, ColumnShapes, InterTableTrends)
from sdmetrics.reports.multi_table.base_multi_table_report import BaseMultiTableReport


class QualityReport(BaseMultiTableReport):
    """Multi table quality report.

    This class creates a quality report for multi-table data. It calculates the quality
    score along three properties - Column Shapes, Column Pair Trends, and Cardinality.
    """

    def __init__(self):
        super().__init__()
        self._properties = {
            'Column Shapes': ColumnShapes(),
            'Column Pair Trends': ColumnPairTrends(),
            'Cardinality': Cardinality(),
            'Intertable Trends': InterTableTrends()
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
