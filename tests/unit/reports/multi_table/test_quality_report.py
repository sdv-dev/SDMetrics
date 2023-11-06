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
