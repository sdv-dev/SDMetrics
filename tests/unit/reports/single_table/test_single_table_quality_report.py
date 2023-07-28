from unittest.mock import Mock, call, patch

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

    @patch('sys.stdout.write')
    def test__print_results(self, mock_write):
        """Test the ``_print_results`` method."""
        # Setup
        quality_report = QualityReport()
        quality_report._overall_score = 0.5
        quality_report._properties = {
            'Column Shapes': Mock(_compute_average=Mock(return_value=0.6)),
            'Column Pair Trends': Mock(_compute_average=Mock(return_value=0.4))
        }

        # Run
        quality_report._print_results()

        # Assert
        calls = [
            call('\nOverall Quality Score: 50.0%\n\n'),
            call('Properties:\n'),
            call('- Column Shapes: 60.0%\n'),
            call('- Column Pair Trends: 40.0%\n'),
        ]
        mock_write.assert_has_calls(calls, any_order=True)

    def test_get_score(self):
        """Test the ``get_score`` method."""
        # Setup
        quality_report = QualityReport()
        mock_score = Mock()
        quality_report._overall_score = mock_score

        # Run
        score = quality_report.get_score()

        # Assert
        assert score == mock_score
