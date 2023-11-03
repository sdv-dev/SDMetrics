from unittest.mock import Mock, patch

from sdmetrics.reports.multi_table import DiagnosticReport
from sdmetrics.reports.multi_table._properties import DataValidity, RelationshipValidity, Structure


class TestDiagnosticReport:

    def test___init__(self):
        """Test that the ``__init__`` method"""
        # Setup
        report = DiagnosticReport()

        # Assert
        assert report._overall_score is None
        assert report.is_generated is False
        assert report.table_names == []
        assert isinstance(report._properties['Data Validity'], DataValidity)
        assert isinstance(report._properties['Data Structure'], Structure)
        assert isinstance(report._properties['Relationship Validity'], RelationshipValidity)

    @patch('sdmetrics.reports.base_report.BaseReport.generate')
    def test_generate_without_verbose(self, mock_super_generate):
        """Test the ``generate`` method without verbose."""
        # Setup
        real_data = Mock()
        synthetic_data = Mock()
        metadata = Mock()
        report = DiagnosticReport()

        # Run
        report.generate(real_data, synthetic_data, metadata, verbose=False)

        # Assert
        mock_super_generate.assert_called_once_with(
            real_data, synthetic_data, metadata, verbose=False)

    def test_get_score(self):
        """Test the ``get_score`` method."""
        # Setup
        report = DiagnosticReport()
        mock_check_report_generated = Mock()
        report._check_report_generated = mock_check_report_generated
        report.is_generated = True
        report._overall_score = 0.7

        # Run
        results = report.get_score()

        # Assert
        mock_check_report_generated.assert_called_once_with()
        assert results == 0.7
