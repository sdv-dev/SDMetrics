from unittest.mock import Mock, patch

from sdmetrics.reports._results_handler import DiagnosticReportResultsHandler
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
        assert isinstance(report._properties['Coverage'], Coverage)
        assert isinstance(report._properties['Boundary'], Boundary)
        assert isinstance(report._properties['Synthesis'], Synthesis)
        assert isinstance(report._results_handler, DiagnosticReportResultsHandler)

    def test__handle_results(self):
        """Test that the proper values are passed to the handler."""
        # Setup
        report = DiagnosticReport()
        report._properties = Mock()
        report._results_handler = Mock()

        # Run
        report._handle_results(True)

        # Assert
        report._results_handler.print_results.assert_called_once_with(
            report._properties, True)

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

    def test_get_results(self):
        """Test the ``get_results`` method."""
        # Setup
        report = DiagnosticReport()
        mock_check_report_generated = Mock()
        report._check_report_generated = mock_check_report_generated
        mock_results_handler = Mock()
        report._results_handler = mock_results_handler
        mock_results_handler.results = {'SUCCESS': ['Test']}
        report.is_generated = True

        # Run
        results = report.get_results()

        # Assert
        mock_check_report_generated.assert_called_once_with()
        assert results == {'SUCCESS': ['Test']}
