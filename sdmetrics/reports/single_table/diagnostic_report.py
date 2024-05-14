"""Single table diagnostic report."""

from sdmetrics.reports.base_report import BaseReport
from sdmetrics.reports.single_table._properties import DataValidity, Structure


class DiagnosticReport(BaseReport):
    """Single table diagnostic report.

    This class creates a diagnostic report for single-table data. It calculates the diagnostic
    score along two properties - Data Structure and Data Validity.
    """

    def __init__(self):
        super().__init__()
        self._properties = {
            'Data Validity': DataValidity(),
            'Data Structure': Structure(),
        }

    def _validate_metadata_matches_data(self, real_data, synthetic_data, metadata):
        return
