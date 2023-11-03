"""Single table diagnostic report."""
import logging

from sdmetrics.reports._results_handler import QualityReportResultsHandler
from sdmetrics.reports.base_report import BaseReport
from sdmetrics.reports.single_table._properties import DataValidity, Structure

LOGGER = logging.getLogger(__name__)


class DiagnosticReport(BaseReport):
    """Single table diagnostic report.

    This class creates a diagnostic report for single-table data. It calculates the diagnostic
    score along three properties - Structure, RelationshipValidity, and DataValidity.
    """

    def __init__(self):
        super().__init__()
        self._properties = {
            'Data Validity': DataValidity(),
            'Data Structure': Structure(),
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
