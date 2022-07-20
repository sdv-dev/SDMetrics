from unittest.mock import Mock, patch

from sdmetrics.column_pairs.base import ColumnPairsMetric


class TestColumnPairsMetric:

    def test_compute_breakdown(self):
        """Test the ``compute_breakdown`` method.

        Expect a breakdown dictionary is returned that contains the score.

        Setup:
        - Mock the ``compute`` method to return a fake score.

        Input:
        - Real data.
        - Synthetic data.

        Output:
        - The evaluated metric.
        """
        # Setup
        metric = ColumnPairsMetric()
        test_metric_score = 0.5

        # Run
        with patch.object(ColumnPairsMetric, 'compute', return_value=test_metric_score):
            result = metric.compute_breakdown(Mock(), Mock())

        # Assert
        assert result == {'score': test_metric_score}
