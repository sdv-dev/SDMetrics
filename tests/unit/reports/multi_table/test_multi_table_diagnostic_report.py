from unittest.mock import Mock, patch

from sdmetrics.reports.multi_table import DiagnosticReport
from sdmetrics.reports.multi_table._properties import Boundary, Coverage, Synthesis


class TestDiagnosticReport:

    def test___init__(self):
        """Test that the ``__init__`` method"""
        # Setup
        report = DiagnosticReport()

        # Assert
        assert report._overall_score is None
        assert report.is_generated is False
        assert report.table_names == []
        assert report.results == {}
        assert isinstance(report._properties['Coverage'], Coverage)
        assert isinstance(report._properties['Boundary'], Boundary)
        assert isinstance(report._properties['Synthesis'], Synthesis)

    @patch('sdmetrics.reports.multi_table.diagnostic_report._print_results_diagnostic_reports')
    def test__print_results(self, mock_print_results):
        """Test the ``_print_results`` method."""
        # Setup
        report = DiagnosticReport()

        # Run
        report._print_results()

        # Assert
        mock_print_results.assert_called_once_with(report)

    @patch('sdmetrics.reports.multi_table.diagnostic_report._generate_results_diagnostic_report')
    @patch('sdmetrics.reports.single_table.base_report.BaseReport.generate')
    def test_generate_without_verbose(self, mock_super_generate, mock_generate_results):
        """Test the ``generate`` method without verbose."""
        # Setup
        real_data = Mock()
        synthetic_data = Mock()
        metadata = Mock()
        report = DiagnosticReport()

        # Run
        report.generate(real_data, synthetic_data, metadata, verbose=False)

        # Assert
        mock_super_generate.assert_called_once_with(real_data, synthetic_data, metadata, False)
        mock_generate_results.assert_called_once_with(report)

    def test_get_results(self):
        """Test the ``get_results`` method."""
        # Setup
        report = DiagnosticReport()
        mock_check_report_generated = Mock()
        report._check_report_generated = mock_check_report_generated
        report.results = {'SUCCESS': ['Test']}
        report.is_generated = True

        # Run
        results = report.get_results()

        # Assert
        mock_check_report_generated.assert_called_once_with()
        assert results == {'SUCCESS': ['Test']}
