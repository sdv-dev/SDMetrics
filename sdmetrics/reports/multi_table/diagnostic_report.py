"""Multi table diagnostic report."""
from sdmetrics.reports._results_handler import QualityReportResultsHandler
from sdmetrics.reports.multi_table._properties import DataValidity, RelationshipValidity, Structure
from sdmetrics.reports.multi_table.base_multi_table_report import BaseMultiTableReport


class DiagnosticReport(BaseMultiTableReport):
    """Multi table diagnostic report.

    This class creates a diagnostic report for multi-table data. It calculates the diagnostic
    score along three properties - RelationshipValidity, DataStructure, and DataValidity.
    """

    def __init__(self):
        super().__init__()
        self._properties = {
            'Data Validity': DataValidity(),
            'Data Structure': Structure(),
            'Relationship Validity': RelationshipValidity()
        }
        self._results_handler = QualityReportResultsHandler()

    def _handle_results(self, verbose):
        self._results_handler.print_results(self._properties, self._overall_score, verbose)

    def get_score(self):
        """Return the diagnostic results.

        Returns:
            dict
                The diagnostic results.
        """
        self._check_report_generated()
        return self._overall_score
