"""Multi table diagnostic report."""

from sdmetrics.reports.multi_table._properties import DataValidity, RelationshipValidity, Structure
from sdmetrics.reports.multi_table.base_multi_table_report import BaseMultiTableReport


class DiagnosticReport(BaseMultiTableReport):
    """Multi table diagnostic report.

    This class creates a diagnostic report for multi-table data. It calculates the diagnostic
    score along three properties - Relationship Validity, Data Structure, and Data Validity.
    """

    def __init__(self):
        super().__init__()
        self._properties = {
            'Data Validity': DataValidity(),
            'Data Structure': Structure(),
            'Relationship Validity': RelationshipValidity(),
        }

    def _validate_metadata_matches_data(self, real_data, synthetic_data, metadata):
        self._validate_relationships(real_data, synthetic_data, metadata)
