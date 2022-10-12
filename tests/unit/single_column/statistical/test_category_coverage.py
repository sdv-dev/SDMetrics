from unittest.mock import Mock, patch

import pandas as pd

from sdmetrics.single_column.statistical import CategoryCoverage


class TestCategoryCoverage:

    def test_compute_breakdown(self):
        """Test the ``compute_breakdown`` method.

        Expect that the number of unique categories is computed for both real and synthetic data.

        Input:
        - Real data.
        - Synthetic data.

        Output:
        - A mapping of the metric results, containing the score and the real and synthetic results.
        """
        # Setup
        real_data = pd.Series(['a', 'b', 'a', 'b', 'c'])
        synthetic_data = pd.Series(['a', 'a', 'a', 'b', 'b'])

        metric = CategoryCoverage()

        # Run
        result = metric.compute_breakdown(real_data, synthetic_data)

        # Assert
        assert result == {'score': 2 / 3, 'real': 3, 'synthetic': 2}

    def test_compute_breakdown_missing_categories(self):
        """Test the ``compute_breakdown`` method with missing categorical values.

        Expect that the number of unique categories is computed for both real and synthetic data.
        """
        # Setup
        real_data = pd.Series(['a', 'b', 'a', 'b', 'c'])
        synthetic_data = pd.Series(['d', 'e', 'f', 'f', 'e'])

        metric = CategoryCoverage()

        # Run
        result = metric.compute_breakdown(real_data, synthetic_data)

        # Assert
        assert result == {'score': 0, 'real': 3, 'synthetic': 0}

    def test_compute(self):
        """Test the ``compute`` method.

        Expect that the number of unique categories is computed for both real and synthetic data.

        Setup:
        - Patch the ``compute_breakdown`` method to return a mapping of the metric results.

        Input:
        - Real data.
        - Synthetic data.

        Output:
        - The evaluated metric.
        """
        # Setup
        metric_breakdown = {'score': 2 / 3, 'real': 3, 'synthetic': 2}

        metric = CategoryCoverage()

        # Run
        with patch.object(CategoryCoverage, 'compute_breakdown', return_value=metric_breakdown):
            result = metric.compute(Mock(), Mock())

        # Assert
        assert result == 2 / 3

    @patch('sdmetrics.single_column.statistical.category_coverage.SingleColumnMetric.normalize')
    def test_normalize(self, normalize_mock):
        """Test the ``normalize`` method.

        Expect that the inherited ``normalize`` method is called.

        Input:
        - Raw score

        Output:
        - The output of the inherited ``normalize`` method.
        """
        # Setup
        metric = CategoryCoverage()
        raw_score = 0.9

        # Run
        result = metric.normalize(raw_score)

        # Assert
        normalize_mock.assert_called_once_with(raw_score)
        assert result == normalize_mock.return_value
