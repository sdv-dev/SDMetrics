from unittest.mock import Mock, patch

import pandas as pd

from sdmetrics.single_column.statistical import StatisticSimilarity


class TestStatisticSimilarity:

    def test_compute_breakdown(self):
        """Test the ``compute_breakdown`` method.

        Expect that the selected statistic method is used to compare the real and synthetic data.

        Setup:
        - Initialize the method with the 'mean' statistic.

        Input:
        - Real data.
        - Synthetic data.

        Output:
        - A mapping of the metric results, containing the score and the real and synthetic results.
        """
        # Setup
        real_data = pd.Series([1.0, 2.4, 2.6, 0.8])
        synthetic_data = pd.Series([0.9, 1.8, 3.1, 5.0])

        metric = StatisticSimilarity(statistic='mean')

        # Run
        result = metric.compute_breakdown(real_data, synthetic_data)

        # Assert
        assert result == {'score': 1 - (2.7 - 1.7) / 1.8, 'real': 1.7, 'synthetic': 2.7}

    def test_compute(self):
        """Test the ``compute`` method.

        Expect that the selected statistic method is used to compare the real and synthetic data.

        Setup:
        - Patch the ``compute_breakdown`` method to return a mapping of the metric results.

        Input:
        - Real data.
        - Synthetic data.

        Output:
        - The evaluated metric.
        """
        # Setup
        metric_breakdown = {'score': 0.56, 'real': 1.7, 'synthetic': 2.7}

        metric = StatisticSimilarity(statistic='mean')

        # Run
        with patch.object(StatisticSimilarity, 'compute_breakdown', return_value=metric_breakdown):
            result = metric.compute(Mock(), Mock())

        # Assert
        assert result == 0.56

    @patch('sdmetrics.single_column.statistical.statistic_similarity.SingleColumnMetric.normalize')
    def test_normalize(self, normalize_mock):
        """Test the ``normalize`` method.

        Expect that the inherited ``normalize`` method is called.

        Input:
        - Raw score

        Output:
        - The output of the inherited ``normalize`` method.
        """
        # Setup
        metric = StatisticSimilarity(statistic='mean')
        raw_score = 0.9

        # Run
        result = metric.normalize(raw_score)

        # Assert
        normalize_mock.assert_called_once_with(raw_score)
        assert result == normalize_mock.return_value
