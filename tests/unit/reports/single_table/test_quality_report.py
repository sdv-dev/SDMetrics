import re

import pytest

from sdmetrics.reports.single_table import QualityReport
from sdmetrics.reports.single_table._properties import ColumnPairTrends, ColumnShapes


class TestQualityReport:
    def test___init__(self):
        """Test the ``__init__`` method."""
        # Run
        expected_warning = re.escape(
            'The single table quality report is deprecated. Please use the QualityReport '
            "from 'sdmetrics.reports' instead."
        )
        with pytest.warns(FutureWarning, match=expected_warning):
            report = QualityReport()

        # Assert
        assert report._overall_score is None
        assert not report.is_generated
        assert isinstance(report._properties['Column Shapes'], ColumnShapes)
        assert isinstance(report._properties['Column Pair Trends'], ColumnPairTrends)
        assert report.real_correlation_threshold == 0.5
        assert report.real_association_threshold == 0.3
