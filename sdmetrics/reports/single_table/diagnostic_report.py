"""Single table diagnostic report."""
import logging
from copy import deepcopy

from sdmetrics.reports._results_handler import DiagnosticReportResultsHandler
from sdmetrics.reports.base_report import BaseReport
from sdmetrics.reports.single_table._properties import Boundary, Coverage, Synthesis

LOGGER = logging.getLogger(__name__)


class DiagnosticReport(BaseReport):
    """Single table diagnostic report.

    This class creates a diagnostic report for single-table data. It calculates the diagnostic
    score along three properties - Synthesis, Coverage, and Boundary.
    """

    def __init__(self):
        super().__init__()
        self._properties = {
            'Coverage': Coverage(),
            'Boundary': Boundary(),
            'Synthesis': Synthesis()
        }
        self._results_handler = DiagnosticReportResultsHandler()

    def _handle_results(self, verbose):
        self._results_handler.print_results(self._properties, verbose)

    def get_results(self):
        """Return the diagnostic results.

        Returns:
            dict
                The diagnostic results.
        """
        self._check_report_generated()
        return deepcopy(self._results_handler.results)
