from unittest.mock import patch

import numpy as np
import pandas as pd
import pytest

from sdmetrics.errors import InvalidDataError
from sdmetrics.single_column.statistical import KeyUniqueness


class TestKeyUniqueness:

    def test_compute_breakdown(self):
        """Test the ``compute_breakdown`` method."""
        # Setup
        real_data = pd.Series([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
        synthetic_data = pd.Series([1, 2, np.nan, 3, np.nan, 5, 2, np.nan, 6, None])

        metric = KeyUniqueness()

        # Run
        result = metric.compute_breakdown(real_data, synthetic_data)

        # Assert
        assert result == {'score': 0.6}

    def test_compute_breakdown_with_duplicates_in_real_data(self):
        """Test the ``compute_breakdown`` method with duplicates in the real data."""
        # Setup
        real_data = pd.Series([1, 2, 3, 4, 5, 6, 7, 8, 9, 10] * 2)
        synthetic_data = pd.Series([1, 2, np.nan, 3, np.nan, 5, 2, np.nan, 6, None])
        metric = KeyUniqueness()

        # Run
        expected_message = 'The real data contains NA or duplicate values.'
        with pytest.raises(InvalidDataError, match=expected_message):
            metric.compute_breakdown(real_data, synthetic_data)

    @patch('sdmetrics.single_column.statistical.key_uniqueness.KeyUniqueness.compute_breakdown')
    def test_compute(self, compute_breakdown_mock):
        """Test the ``compute`` method."""
        # Setup
        real_data = pd.Series([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
        synthetic_data = pd.Series([1, 2, np.nan, 3, np.nan, 5, 2, np.nan, 6, None])
        metric = KeyUniqueness()
        compute_breakdown_mock.return_value = {'score': 0.6}

        # Run
        result = metric.compute(real_data, synthetic_data)

        # Assert
        assert result == 0.6

    @patch('sdmetrics.single_column.statistical.key_uniqueness.SingleColumnMetric.normalize')
    def test_normalize(self, normalize_mock):
        """Test the ``normalize`` method."""
        # Setup
        metric = KeyUniqueness()
        raw_score = 0.9

        # Run
        result = metric.normalize(raw_score)

        # Assert
        normalize_mock.assert_called_once_with(raw_score)
        assert result == normalize_mock.return_value
