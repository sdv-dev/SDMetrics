"""Single table diagnostic report."""
import logging

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
