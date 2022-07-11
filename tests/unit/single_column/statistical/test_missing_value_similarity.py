from unittest.mock import Mock, patch

import numpy as np
import pandas as pd

from sdmetrics.single_column.statistical import MissingValueSimilarity


class TestMissingValueSimilarity:

    def test_compute_breakdown(self):
        """Test the ``compute_breakdown`` method.

        Expect that the number of missing values is computed for both real and synthetic data.

        Input:
        - Real data.
        - Synthetic data.

        Output:
        - A mapping of the metric results, containing the score and the real and synthetic results.
        """
        # Setup
        real_data = pd.Series([1.0, np.nan, 2.6, 0.8])
        synthetic_data = pd.Series([0.9, 1.8, None, None])

        metric = MissingValueSimilarity()

        # Run
        result = metric.compute_breakdown(real_data, synthetic_data)

        # Assert
        assert result == {'score': 0.75, 'real': 0.25, 'synthetic': 0.5}

    def test_compute(self):
        """Test the ``compute`` method.

        Expect that the number of missing values is computed for both real and synthetic data.

        Setup:
        - Patch the ``compute_breakdown`` method to return a mapping of the metric results.

        Input:
        - Real data.
        - Synthetic data.

        Output:
        - The evaluated metric.
        """
        # Setup
        metric_breakdown = {'score': 0.75, 'real': 0.25, 'synthetic': 0.75}

        metric = MissingValueSimilarity()

        # Run
        with patch.object(
            MissingValueSimilarity,
            'compute_breakdown',
            return_value=metric_breakdown,
        ):
            result = metric.compute(Mock(), Mock())

        # Assert
        assert result == 0.75

    @patch(
        'sdmetrics.single_column.statistical.missing_value_similarity.SingleColumnMetric.normalize'
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
        metric = MissingValueSimilarity()
        raw_score = 0.9

        # Run
        result = metric.normalize(raw_score)

        # Assert
        normalize_mock.assert_called_once_with(raw_score)
        assert result == normalize_mock.return_value
