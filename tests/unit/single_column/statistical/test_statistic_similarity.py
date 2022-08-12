from unittest.mock import Mock, patch

import numpy as np
import pandas as pd

from sdmetrics.single_column.statistical import StatisticSimilarity
from sdmetrics.warnings import ConstantInputWarning


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

        metric = StatisticSimilarity()

        # Run
        result = metric.compute_breakdown(real_data, synthetic_data, statistic='mean')

        # Assert
        assert result == {'score': 1 - (2.7 - 1.7) / 1.8, 'real': 1.7, 'synthetic': 2.7}

    def test_compute_breakdown_constant_input(self):
        """Test the ``compute_breakdown`` method with constant input.

        Expect that an invalid score is returned and that a warning is thrown.

        Input:
        - Mocked real data.
        - Mocked synthetic data.

        Output:
        - A mapping of the metric results, containing the score and the real and synthetic results.
        """
        # Setup
        real_data = pd.Series([1.0, 1.0, 1.0])
        synthetic_data = pd.Series([0.9, 1.8, 3.1])
        expected_score_breakdown = {
            'score': np.nan,
        }
        expected_warn_msg = (
            'The real data input array is constant. '
            'The StatisticSimilarity metric is either undefined or infinte.'
        )

        # Run
        metric = StatisticSimilarity()
        with np.testing.assert_warns(ConstantInputWarning, match=expected_warn_msg):
            result = metric.compute_breakdown(real_data, synthetic_data, statistic='mean')

        # Assert
        assert result == expected_score_breakdown

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

        metric = StatisticSimilarity()

        # Run
        with patch.object(StatisticSimilarity, 'compute_breakdown', return_value=metric_breakdown):
            result = metric.compute(Mock(), Mock(), statistic='mean')

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
        metric = StatisticSimilarity()
        raw_score = 0.9

        # Run
        result = metric.normalize(raw_score)

        # Assert
        normalize_mock.assert_called_once_with(raw_score)
        assert result == normalize_mock.return_value
