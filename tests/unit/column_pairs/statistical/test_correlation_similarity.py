from unittest.mock import Mock, patch

import pandas as pd

from sdmetrics.column_pairs.statistical import CorrelationSimilarity
from tests.utils import SeriesMatcher


class TestCorrelationSimilarity:

    @patch('sdmetrics.column_pairs.statistical.correlation_similarity.pearsonr')
    def test_compute(self, pearson_mock):
        """Test the ``compute`` method.

        Expect that the selected coefficient is used to compare the real and synthetic data.

        Setup:
        - Patch the ``scipy.stats.pearsonr`` method to return a test result.

        Input:
        - Real data.
        - Synthetic data.

        Output:
        - The evaluated metric.
        """
        # Setup
        real_data = pd.Series([1.0, 2.4, 2.6, 0.8])
        synthetic_data = pd.Series([0.9, 1.8, 3.1, 5.0])

        metric = CorrelationSimilarity()

        # Run
        result = metric.compute(real_data, synthetic_data, coefficient='Pearson')

        # Assert
        pearson_mock.assert_called_once_with(
            SeriesMatcher(real_data), SeriesMatcher(synthetic_data))
        assert result == pearson_mock.return_value

    def test_compute_breakdown(self):
        """Test the ``compute_breakdown`` method.

        Expect that the selected coefficient is used to compare the real and synthetic data.

        Setup:
        - Mock the ``compute`` method to return a test score.

        Input:
        - Mocked real data.
        - Mocked synthetic data.

        Output:
        - A mapping of the metric results, containing the score and the real and synthetic results.
        """
        # Setup
        test_score = 0.2
        metric = CorrelationSimilarity()

        # Run
        with patch.object(CorrelationSimilarity, 'compute', return_value=test_score):
            result = metric.compute_breakdown(Mock(), Mock(), coefficient='Pearson')

        # Assert
        assert result == {'score': test_score}

    @patch(
        'sdmetrics.column_pairs.statistical.correlation_similarity.ColumnPairsMetric.normalize'
    )
    def test_normalize(self, normalize_mock):
        """Test the ``normalize`` method.

        Expect that the inherited ``normalize`` method is called.

        Input:
        - Raw score

        Output:
        - The output of the inherited ``normalize`` method.
        """
        # Setup
        metric = CorrelationSimilarity()
        raw_score = 0.9

        # Run
        result = metric.normalize(raw_score)

        # Assert
        normalize_mock.assert_called_once_with(raw_score)
        assert result == normalize_mock.return_value
