from unittest.mock import Mock, patch

from sdmetrics.reports.multi_table import QualityReport
from sdmetrics.reports.multi_table._properties import Cardinality, ColumnPairTrends, ColumnShapes


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

    @patch('sdmetrics.reports.multi_table.quality_report._print_results_quality_report')
    def test__print_results(self, mock_print_result):
        """Test the ``_print_results`` method."""
        # Setup
        report = QualityReport()

        # Run
        report._print_results()

        # Assert
        mock_print_result.assert_called_once_with(report)

    @patch('sdmetrics.reports.single_table.base_report.BaseReport.generate')
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
