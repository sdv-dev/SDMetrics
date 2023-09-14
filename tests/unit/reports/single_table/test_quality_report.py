from unittest.mock import Mock, patch

from sdmetrics.reports._results_handler import QualityReportResultsHandler
from sdmetrics.reports.single_table import QualityReport
from sdmetrics.reports.single_table._properties import ColumnPairTrends, ColumnShapes


class TestQualityReport:

    def test___init__(self):
        """Test the ``__init__`` method."""
        # Run
        report = QualityReport()

        # Assert
        assert report._overall_score is None
        assert not report.is_generated
        assert isinstance(report._properties['Column Shapes'], ColumnShapes)
        assert isinstance(report._properties['Column Pair Trends'], ColumnPairTrends)
        assert isinstance(report._results_handler, QualityReportResultsHandler)

    @patch('sys.stdout.write')
    def test__handle_results(self, mock_write):
        """Test that the proper values are passed to the handler."""
        # Setup
        quality_report = QualityReport()
        quality_report._overall_score = 0.5
        quality_report._properties = {
            'Column Shapes': Mock(_compute_average=Mock(return_value=0.6)),
            'Column Pair Trends': Mock(_compute_average=Mock(return_value=0.4))
        }
        quality_report._results_handler = Mock()

        # Run
        quality_report._handle_results(True)

        # Assert
        quality_report._results_handler.print_results.assert_called_once_with(
            quality_report._properties, quality_report._overall_score, True)

    def test_get_score(self):
        """Test the ``get_score`` method."""
        # Setup
        quality_report = QualityReport()
        quality_report.is_generated = True
        mock_score = Mock()
        quality_report._overall_score = mock_score

        # Run
        score = quality_report.get_score()

        # Assert
        assert score == mock_score
