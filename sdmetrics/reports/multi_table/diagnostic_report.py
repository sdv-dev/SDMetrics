"""Multi table diagnostic report."""
from copy import deepcopy

from sdmetrics.reports.multi_table._properties import Boundary, Coverage, Synthesis
from sdmetrics.reports.multi_table.base_multi_table_report import BaseMultiTableReport
from sdmetrics.reports.utils import (
    _generate_results_diagnostic_report, _print_results_diagnostic_reports)


class DiagnosticReport(BaseMultiTableReport):
    """Multi table diagnostic report.

    This class creates a diagnostic report for multi-table data. It calculates the diagnostic
    score along three properties - Synthesis, Coverage, and Boundary.
    """

    def __init__(self):
        super().__init__()
        self._properties = {
            'Coverage': Coverage(),
            'Boundary': Boundary(),
            'Synthesis': Synthesis()
        }
        self.results = {}

    def _print_results(self):
        """Print the results of the report."""
        _print_results_diagnostic_reports(self)

    def generate(self, real_data, synthetic_data, metadata, verbose=True):
        """Generate report.

        Args:
            real_data (pandas.DataFrame):
                The real data.
            synthetic_data (pandas.DataFrame):
                The synthetic data.
            metadata (dict):
                The metadata, which contains each column's data type.
            verbose (bool):
                Whether or not to print report summary and progress.
        """
        super().generate(real_data, synthetic_data, metadata, verbose)
        if not verbose:
            _generate_results_diagnostic_report(self)

    def get_results(self):
        """Return the diagnostic results.

        Returns:
            dict
                The diagnostic results.
        """
        self._check_report_generated()
        return deepcopy(self.results)
