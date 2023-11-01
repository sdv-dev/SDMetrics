from unittest.mock import Mock, patch

from sdmetrics.reports._results_handler import QualityReportResultsHandler
from sdmetrics.reports.multi_table import QualityReport
from sdmetrics.reports.multi_table._properties import (
    Cardinality, ColumnPairTrends, ColumnShapes, InterTableTrends)


class TestQualityReport:

    def test___init__(self):
        """Test that the ``__init__`` method"""
        # Setup
        report = QualityReport()

        # Assert
        assert report._overall_score is None
        assert report.is_generated is False
        assert report.table_names == []
        assert isinstance(report._properties['Column Shapes'], ColumnShapes)
        assert isinstance(report._properties['Column Pair Trends'], ColumnPairTrends)
        assert isinstance(report._properties['Cardinality'], Cardinality)
        assert isinstance(report._properties['Intertable Trends'], InterTableTrends)
        assert isinstance(report._results_handler, QualityReportResultsHandler)

    def test__handle_results(self):
        """Test that the proper values are passed to the handler."""
        # Setup
        report = QualityReport()
        report._overall_score = 0.5
        report._properties = {
            'Column Shapes': Mock(_compute_average=Mock(return_value=0.6)),
            'Column Pair Trends': Mock(_compute_average=Mock(return_value=0.4))
        }
        report._results_handler = Mock()

        # Run
        report._handle_results(True)

        # Assert
        report._results_handler.print_results.assert_called_once_with(
            report._properties, report._overall_score, True)

    @patch('sdmetrics.reports.base_report.BaseReport.generate')
    def test_generate_without_verbose(self, mock_super_generate):
        """Test the ``generate`` method without verbose."""
        # Setup
        real_data = Mock()
        synthetic_data = Mock()
        metadata = Mock()
        report = QualityReport()

        # Run
        report.generate(real_data, synthetic_data, metadata, verbose=False)

        # Assert
        mock_super_generate.assert_called_once_with(
            real_data, synthetic_data, metadata, verbose=False
        )

    def test_get_score(self):
        """Test the ``get_score`` method."""
        # Setup
        report = QualityReport()
        mock_check_report_generated = Mock()
        report._check_report_generated = mock_check_report_generated
        report._overall_score = 0.5
        report.is_generated = True

        # Run
        score = report.get_score()

        # Assert
        assert score == 0.5
        mock_check_report_generated.assert_called_once_with()
