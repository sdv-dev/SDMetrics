from unittest.mock import patch

import numpy as np
import pandas as pd

from sdmetrics.single_column.statistical import RangeCoverage


class TestRangeCoverage:

    def test_compute(self):
        """Test the ``compute`` method.

        Expect that the range coverage is returned.

        Input:
        - Real data.
        - Synthetic data.

        Output:
        - The evaluated metric.
        """
        # Setup
        real_data = pd.Series([1, 2, 3, 4, 5])
        synthetic_data = pd.Series([2, 3, 1, 2, 3])

        metric = RangeCoverage()

        # Run
        result = metric.compute(real_data, synthetic_data)

        # Assert
        assert result == 0.5

    def test_compute_constant_input(self):
        """Test the ``compute`` method with constant input.

        Expect that the range coverage is returned.

        Input:
        - Real data.
        - Synthetic data.

        Output:
        - The evaluated metric.
        """
        # Setup
        real_data = pd.Series([1, 1, 1, 1, 1])
        synthetic_data = pd.Series([2, 3, 1, 2, 3])

        metric = RangeCoverage()

        # Run
        result = metric.compute(real_data, synthetic_data)

        # Assert
        assert np.isnan(result)

    def test_compute_with_nans(self):
        """Test the ``compute`` method with NaN values in the input data.

        Expect that the range coverage is returned.
        """
        # Setup
        real_data = pd.Series([1, np.nan, 3, 4, 5])
        synthetic_data = pd.Series([2, 3, 1, np.nan, 3])

        metric = RangeCoverage()

        # Run
        result = metric.compute(real_data, synthetic_data)

        # Assert
        assert result == 0.5

    def test_compute_with_list_input(self):
        """Test the ``compute`` method with list input.

        Expect that the range coverage is returned.
        """
        # Setup
        real_data = [1, np.nan, 3, 4, 5]
        synthetic_data = [2, 3, 1, np.nan, 3]

        metric = RangeCoverage()

        # Run
        result = metric.compute(real_data, synthetic_data)

        # Assert
        assert result == 0.5

    @patch('sdmetrics.single_column.statistical.range_coverage.SingleColumnMetric.normalize')
    def test_normalize(self, normalize_mock):
        """Test the ``normalize`` method.

        Expect that the inherited ``normalize`` method is called.

        Input:
        - Raw score

        Output:
        - The output of the inherited ``normalize`` method.
        """
        # Setup
        metric = RangeCoverage()
        raw_score = 0.9

        # Run
        result = metric.normalize(raw_score)

        # Assert
        normalize_mock.assert_called_once_with(raw_score)
        assert result == normalize_mock.return_value
