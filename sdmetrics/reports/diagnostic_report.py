"""Unified diagnostic report."""

from sdmetrics.reports.base_unified_report import BaseUnifiedReport
from sdmetrics.reports.multi_table._properties import DataValidity, RelationshipValidity, Structure


class DiagnosticReport(BaseUnifiedReport):
    """Diagnostic report for single-table and multi-table data.

    This class creates a diagnostic report for single-table data.
    It calculates the quality score using the multi-table report properties.
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
