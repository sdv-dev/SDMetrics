import re

import pytest

from sdmetrics.reports import DiagnosticReport
from sdmetrics.reports.multi_table import DiagnosticReport as MultiTableDiagnosticReport
from sdmetrics.reports.multi_table._properties import DataValidity, RelationshipValidity, Structure


class TestDiagnosticReport:
    def test___init__(self):
        """Test that the ``__init__`` method for MultiTableDiagnosticReport."""
        # Setup and Run
        expected_warning = re.escape(
            'The multi table diagnostic report is deprecated. Please use the DiagnosticReport '
            "from 'sdmetrics.reports' instead."
        )
        with pytest.warns(FutureWarning, match=expected_warning):
            report = MultiTableDiagnosticReport()

        # Assert
        assert report._overall_score is None
        assert report.is_generated is False
        assert report.table_names == []
        assert isinstance(report._properties['Data Validity'], DataValidity)
        assert isinstance(report._properties['Data Structure'], Structure)
        assert isinstance(report._properties['Relationship Validity'], RelationshipValidity)

    def test___init__unified(self):
        """Test that the ``__init__`` method for DiagnosticReport."""
        # Setup and Run
        report = DiagnosticReport()

        # Assert
        assert report._overall_score is None
        assert report.is_generated is False
        assert report.table_names == []
        assert isinstance(report._properties['Data Validity'], DataValidity)
        assert isinstance(report._properties['Data Structure'], Structure)
        assert isinstance(report._properties['Relationship Validity'], RelationshipValidity)
