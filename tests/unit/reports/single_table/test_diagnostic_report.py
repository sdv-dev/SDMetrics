import re

import pytest

from sdmetrics.reports.single_table import DiagnosticReport
from sdmetrics.reports.single_table._properties import DataValidity, Structure


class TestDiagnosticReport:
    def test___init__(self):
        """Test the ``__init__`` method."""
        # Run
        expected_warning = re.escape(
            "The single table diagnostic report is deprecated. Please use the DiagnosticReport "
            "from 'sdmetrics.reports' instead."
        )
        with pytest.warns(FutureWarning, match=expected_warning):
            report = DiagnosticReport()

        # Assert
        assert report._overall_score is None
        assert report.is_generated is False
        assert isinstance(report._properties['Data Validity'], DataValidity)
        assert isinstance(report._properties['Data Structure'], Structure)
