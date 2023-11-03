from unittest.mock import Mock, patch

from sdmetrics.reports._results_handler import QualityReportResultsHandler
from sdmetrics.reports.single_table import DiagnosticReport
from sdmetrics.reports.single_table._properties import DataValidity, Structure


class TestDiagnosticReport:

    def test___init__(self):
        """Test the ``__init__`` method."""
        # Run
        report = DiagnosticReport()

        # Assert
        assert report._overall_score is None
        assert report.is_generated is False
        assert isinstance(report._properties['Data Validity'], DataValidity)
        assert isinstance(report._properties['Data Structure'], Structure)
        assert isinstance(report._results_handler, QualityReportResultsHandler)

    def test__handle_results(self):
        """Test that the proper values are passed to the handler."""
        # Setup
        report = DiagnosticReport()
        report._properties = Mock()
        report._results_handler = Mock()
        report._overall_score = 0.7

        # Run
        report._handle_results(True)

        # Assert
        report._results_handler.print_results.assert_called_once_with(
            report._properties, 0.7, True
        )

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
        report._check_report_generated = Mock()
        report._overall_score = 0.7
        report.is_generated = True

        # Run
        results = report.get_score()

        # Assert
        report._check_report_generated.assert_called_once_with()
        assert results == 0.7
