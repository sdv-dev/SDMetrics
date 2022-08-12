from datetime import datetime
from unittest.mock import Mock, call, patch

import numpy as np
import pandas as pd

from sdmetrics.column_pairs.statistical import CorrelationSimilarity
from sdmetrics.warnings import ConstantInputWarning
from tests.utils import SeriesMatcher


class TestCorrelationSimilarity:

    @patch('sdmetrics.column_pairs.statistical.correlation_similarity.pearsonr')
    def test_compute_breakdown(self, pearson_mock):
        """Test the ``compute_breakdown`` method.

        Expect that the selected coefficient is used to compare the real and synthetic data.

        Setup:
        - Patch the ``scipy.stats.pearsonr`` method to return a test result.

        Input:
        - Mocked real data.
        - Mocked synthetic data.

        Output:
        - A mapping of the metric results, containing the score and the real and synthetic results.
        """
        # Setup
        real_data = pd.DataFrame({'col1': [1.0, 2.4, 2.6, 0.8], 'col2': [1, 2, 3, 4]})
        synthetic_data = pd.DataFrame({'col1': [0.9, 1.8, 3.1, 5.0], 'col2': [2, 3, 4, 1]})
        score_real = -0.451
        score_synthetic = -0.003
        pearson_mock.side_effect = [(score_real, 0.1), (score_synthetic, 0.1)]
        expected_score_breakdown = {
            'score': 1 - abs(score_real - score_synthetic) / 2,
            'real': score_real,
            'synthetic': score_synthetic,
        }

        # Run
        metric = CorrelationSimilarity()
        result = metric.compute_breakdown(real_data, synthetic_data, coefficient='Pearson')

        # Assert
        assert pearson_mock.has_calls(
            call(SeriesMatcher(real_data['col1']), SeriesMatcher(real_data['col2'])),
            call(SeriesMatcher(synthetic_data['col1']), SeriesMatcher(synthetic_data['col2'])),
        )
        assert result == expected_score_breakdown

    @patch('sdmetrics.column_pairs.statistical.correlation_similarity.pearsonr')
    def test_compute_breakdown_datetime(self, pearson_mock):
        """Test the ``compute_breakdown`` method with datetime input.

        Expect that the selected coefficient is used to compare the real and synthetic data.

        Setup:
        - Patch the ``scipy.stats.pearsonr`` method to return a test result.

        Input:
        - Mocked real data.
        - Mocked synthetic data.

        Output:
        - A mapping of the metric results, containing the score and the real and synthetic results.
        """
        # Setup
        real_data = pd.DataFrame({
            'col1': [datetime(2020, 1, 3), datetime(2020, 10, 13), datetime(2021, 5, 3)],
            'col2': [datetime(2021, 7, 23), datetime(2021, 8, 3), datetime(2020, 9, 24)],
        })
        synthetic_data = pd.DataFrame({
            'col1': [datetime(2021, 9, 19), datetime(2021, 10, 1), datetime(2020, 3, 1)],
            'col2': [datetime(2022, 4, 28), datetime(2021, 7, 31), datetime(2020, 4, 2)],
        })
        score_real = 0.2
        score_synthetic = 0.1
        pearson_mock.side_effect = [(score_real, 0.1), (score_synthetic, 0.2)]
        expected_score_breakdown = {
            'score': 1 - abs(score_real - score_synthetic) / 2,
            'real': score_real,
            'synthetic': score_synthetic,
        }

        # Run
        metric = CorrelationSimilarity()
        result = metric.compute_breakdown(real_data, synthetic_data, coefficient='Pearson')

        # Assert
        assert pearson_mock.has_calls(
            call(SeriesMatcher(real_data['col1']), SeriesMatcher(real_data['col2'])),
            call(SeriesMatcher(synthetic_data['col1']), SeriesMatcher(synthetic_data['col2'])),
        )
        assert result == expected_score_breakdown

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
        real_data = pd.DataFrame({'col1': [1.0, 1.0, 1.0], 'col2': [2.0, 2.0, 2.0]})
        synthetic_data = pd.DataFrame({'col1': [0.9, 1.8, 3.1], 'col2': [2, 3, 4]})
        expected_score_breakdown = {
            'score': np.nan,
        }
        expected_warn_msg = (
            'One or both of the input arrays is constant. '
            'The CorrelationSimilarity metric is either undefined or infinte.'
        )

        # Run
        metric = CorrelationSimilarity()
        with np.testing.assert_warns(ConstantInputWarning, match=expected_warn_msg):
            result = metric.compute_breakdown(real_data, synthetic_data, coefficient='Pearson')

        # Assert
        assert result == expected_score_breakdown

    def test_compute(self):
        """Test the ``compute`` method.

        Expect that the selected coefficient is used to compare the real and synthetic data.

        Setup:
        - Mock the ``compute`` method to return a test score.

        Input:
        - Real data.
        - Synthetic data.

        Output:
        - The evaluated metric.
        """
        # Setup
        test_score = 0.2
        score_breakdown = {'score': test_score}
        metric = CorrelationSimilarity()

        # Run
        with patch.object(
            CorrelationSimilarity,
            'compute_breakdown',
            return_value=score_breakdown,
        ):
            result = metric.compute(Mock(), Mock(), coefficient='Pearson')

        # Assert
        assert result == test_score

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
