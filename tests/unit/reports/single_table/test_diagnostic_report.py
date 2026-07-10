import re
import sys

import pytest

from sdmetrics.reports.single_table._properties import DataValidity, Structure


class TestDiagnosticReport:
    def test_warns_on_import(self):
        """Test that importing the deprecated report produces a warning."""
        # Setup
        expected_warning = re.escape(
            'The single table diagnostic report is deprecated. Please use the DiagnosticReport '
            "from 'sdmetrics.reports' instead."
        )
        sys.modules.pop('sdmetrics.reports.single_table.diagnostic_report', None)

        # Run and Assert
        with pytest.warns(FutureWarning, match=expected_warning):
            from sdmetrics.reports.single_table import DiagnosticReport
        assert DiagnosticReport.__name__ == 'DiagnosticReport'

    @pytest.mark.filterwarnings(
        'ignore:The single table diagnostic report is deprecated:FutureWarning'
    )
    def test___init__(self):
        """Test the ``__init__`` method."""
        # Run
        from sdmetrics.reports.single_table import DiagnosticReport

        report = DiagnosticReport()

        # Assert
        assert report._overall_score is None
        assert report.is_generated is False
        assert isinstance(report._properties['Data Validity'], DataValidity)
        assert isinstance(report._properties['Data Structure'], Structure)
