from unittest.mock import Mock, patch

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
