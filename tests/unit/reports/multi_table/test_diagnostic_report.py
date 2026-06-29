import re
import sys

import pytest

from sdmetrics.reports import DiagnosticReport
from sdmetrics.reports.multi_table._properties import DataValidity, RelationshipValidity, Structure


class TestDiagnosticReport:
    def test_warns_on_import(self):
        """Test that importing the deprecated report produces a warning."""
        # Setup
        expected_warning = re.escape(
            'The multi table diagnostic report is deprecated. Please use the DiagnosticReport '
            "from 'sdmetrics.reports' instead."
        )
        sys.modules.pop('sdmetrics.reports.multi_table.diagnostic_report', None)

        # Run and Assert
        with pytest.warns(FutureWarning, match=expected_warning):
            from sdmetrics.reports.multi_table import DiagnosticReport as MultiTableDiagnosticReport
        assert MultiTableDiagnosticReport.__name__ == 'DiagnosticReport'

    @pytest.mark.filterwarnings(
        'ignore:The multi table diagnostic report is deprecated:FutureWarning'
    )
    def test___init__(self):
        """Test that the ``__init__`` method for MultiTableDiagnosticReport."""
        # Setup and Run
        from sdmetrics.reports.multi_table import DiagnosticReport as MultiTableDiagnosticReport

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
