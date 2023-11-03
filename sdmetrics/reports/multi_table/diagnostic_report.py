"""Multi table diagnostic report."""
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

    def get_score(self):
        """Return the diagnostic results.

        Returns:
            dict
                The diagnostic results.
        """
        self._check_report_generated()
        return self._overall_score
