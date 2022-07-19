from unittest.mock import patch

import pandas as pd

from sdmetrics.column_pairs.statistical import ContingencySimilarity


class TestContingencySimilarity:

    def test_compute(self):
        """Test the ``compute`` method.

        Expect that the total variation distance of the two contingency matricies
        is computed.

        Input:
        - Real data.
        - Synthetic data.

        Output:
        - The metric result.
        """
        # Setup
        real_data = pd.DataFrame({'col1': [1.0, 2.4, 2.6, 0.8], 'col2': [1, 2, 3, 5]})
        synthetic_data = pd.DataFrame({'col1': [1.0, 1.8, 2.6, 1.0], 'col2': [2, 3, 4, 1]})
        expected_score = 0.25

        # Run
        metric = ContingencySimilarity()
        result = metric.compute(real_data, synthetic_data)

        # Assert
        assert result == expected_score

    @patch(
        'sdmetrics.column_pairs.statistical.contingency_similarity.ColumnPairsMetric.normalize'
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
        metric = ContingencySimilarity()
        raw_score = 0.9

        # Run
        result = metric.normalize(raw_score)

        # Assert
        normalize_mock.assert_called_once_with(raw_score)
        assert result == normalize_mock.return_value
