import re
import sys

import pytest

from sdmetrics.reports import QualityReport
from sdmetrics.reports.multi_table._properties import (
    Cardinality,
    ColumnPairTrends,
    ColumnShapes,
    InterTableTrends,
)


class TestQualityReport:
    def test_warns_on_import(self):
        """Test that importing the deprecated report produces a warning."""
        # Setup
        expected_warning = re.escape(
            'The multi table quality report is deprecated. Please use the QualityReport '
            "from 'sdmetrics.reports' instead."
        )
        sys.modules.pop('sdmetrics.reports.multi_table.quality_report', None)

        # Run and Assert
        with pytest.warns(FutureWarning, match=expected_warning):
            from sdmetrics.reports.multi_table import QualityReport as MultiTableQualityReport
        assert MultiTableQualityReport.__name__ == 'QualityReport'

    @pytest.mark.filterwarnings('ignore:The multi table quality report is deprecated:FutureWarning')
    def test___init__(self):
        """Test that the ``__init__`` method for MultiTableQualityReport."""
        # Setup and Run
        from sdmetrics.reports.multi_table import QualityReport as MultiTableQualityReport

        report = MultiTableQualityReport()

        # Assert
        assert report._overall_score is None
        assert report.is_generated is False
        assert report.table_names == []
        assert isinstance(report._properties['Column Shapes'], ColumnShapes)
        assert isinstance(report._properties['Column Pair Trends'], ColumnPairTrends)
        assert isinstance(report._properties['Cardinality'], Cardinality)
        assert isinstance(report._properties['Intertable Trends'], InterTableTrends)
        assert report.real_correlation_threshold == 0.5
        assert report.real_association_threshold == 0.3

    def test___init__unified(self):
        """Test that the ``__init__`` method for QualityReport."""
        # Setup and Run
        report = QualityReport()

        # Assert
        assert report._overall_score is None
        assert report.is_generated is False
        assert report.table_names == []
        assert isinstance(report._properties['Column Shapes'], ColumnShapes)
        assert isinstance(report._properties['Column Pair Trends'], ColumnPairTrends)
        assert isinstance(report._properties['Cardinality'], Cardinality)
        assert isinstance(report._properties['Intertable Trends'], InterTableTrends)
        assert report.real_correlation_threshold == 0.5
        assert report.real_association_threshold == 0.3
