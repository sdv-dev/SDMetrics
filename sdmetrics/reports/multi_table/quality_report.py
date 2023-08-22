"""Multi table quality report."""
from sdmetrics.reports.multi_table._properties import Cardinality, ColumnPairTrends, ColumnShapes
from sdmetrics.reports.multi_table.base_multi_table_report import BaseMultiTableReport
from sdmetrics.reports.utils import _print_results_quality_report


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
            'Cardinality': Cardinality()
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
