import re
from datetime import datetime
from unittest.mock import Mock, call, patch

import numpy as np
import pandas as pd
import pytest

from sdmetrics.column_pairs.statistical import CorrelationSimilarity
from sdmetrics.errors import ConstantInputError
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
        pearson_mock.assert_has_calls([
            call(SeriesMatcher(real_data['col1']), SeriesMatcher(real_data['col2'])),
            call(SeriesMatcher(synthetic_data['col1']), SeriesMatcher(synthetic_data['col2'])),
        ])
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
        pearson_mock.assert_has_calls([
            call(
                SeriesMatcher(real_data['col1'].astype('int64')),
                SeriesMatcher(real_data['col2'].astype('int64')),
            ),
            call(
                SeriesMatcher(synthetic_data['col1'].astype('int64')),
                SeriesMatcher(synthetic_data['col2'].astype('int64')),
            ),
        ])
        assert result == expected_score_breakdown

    def test_compute_breakdown_constant_input(self):
        """Test an error is thrown when constant data is passed."""
        # Setup
        real_data = pd.DataFrame({'col1': [1.0, 1.0, 1.0], 'col2': [2.0, 2.0, 2.0]})
        synthetic_data = pd.DataFrame({'col1': [0.9, 1.8, 3.1], 'col2': [2, 3, 4]})

        # Run and Assert
        error_msg = (
            "The real data in columns 'col1, col2' contains a constant value. "
            'Correlation is undefined for constant data.'
        )
        metric = CorrelationSimilarity()
        with pytest.raises(ConstantInputError, match=error_msg):
            metric.compute_breakdown(real_data, synthetic_data, coefficient='Pearson')

    @pytest.mark.parametrize(
        'real_correlation_threshold, score',
        [
            (0, 0.9008941765855183),
            (0.35, 0.9008941765855183),
            (0.498212, np.nan),
            (0.75, np.nan),
        ],
    )
    def test_compute_breakdown_with_real_correlation_threshold(
        self, real_correlation_threshold, score
    ):
        """Test the ``compute_breakdown`` method with real correlation threshold.

        In this test, real data has a correlation of 0.498212 and synthetic data
        has a correlation of 0.3.
        """
        # Setup
        real_data = pd.DataFrame({
            'col1': [1.0, 2.0, 3.0, 4.0],
            'col2': [0.2, -1.0895, -0.6425, 1.5365],
        })
        synthetic_data = pd.DataFrame({
            'col1': [1.0, 2.0, 3.0, 4.0],
            'col2': [0.616536, -1.216536, -0.916536, 1.516536],
        })

        # Run
        metric = CorrelationSimilarity()
        result = metric.compute_breakdown(
            real_data,
            synthetic_data,
            coefficient='Pearson',
            real_correlation_threshold=real_correlation_threshold,
        )

        # Assert
        assert result['score'] == score if not np.isnan(score) else np.isnan(result['score'])

    def test_compute_breakdown_invalid_real_correlation_threshold(self):
        """Test an error is thrown when an invalid `real_correlation_threshold` is passed."""
        # Setup
        real_data = pd.DataFrame({'col1': [1.0, 2.0, 3.0], 'col2': [2.0, 3.0, 4.0]})
        synthetic_data = pd.DataFrame({'col1': [0.9, 1.8, 3.1], 'col2': [2, 3, 4]})
        expected_error = re.escape('real_correlation_threshold must be a number between 0 and 1.')
        metric = CorrelationSimilarity()

        # Run and Assert
        with pytest.raises(ValueError, match=expected_error):
            metric.compute_breakdown(
                real_data,
                synthetic_data,
                coefficient='Pearson',
                real_correlation_threshold=-0.1,
            )

        with pytest.raises(ValueError, match=expected_error):
            metric.compute_breakdown(
                real_data,
                synthetic_data,
                coefficient='Pearson',
                real_correlation_threshold=None,
            )

    @patch(
        'sdmetrics.column_pairs.statistical.correlation_similarity.CorrelationSimilarity.compute_breakdown'
    )
    def test_compute(self, compute_breakdown_mock):
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
        compute_breakdown_mock.return_value = score_breakdown
        real_data = Mock()
        synthetic_data = Mock()

        # Run
        result = metric.compute(
            real_data, synthetic_data, coefficient='Pearson', real_correlation_threshold=0.6
        )

        # Assert
        assert result == test_score
        compute_breakdown_mock.assert_called_once_with(
            real_data,
            synthetic_data,
            'Pearson',
            0.6,
        )

    @patch('sdmetrics.column_pairs.statistical.correlation_similarity.ColumnPairsMetric.normalize')
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
