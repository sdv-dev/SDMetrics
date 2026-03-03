from unittest.mock import patch

import numpy as np
import pandas as pd
import pytest

from sdmetrics.single_column.statistical import KeyUniqueness


class TestKeyUniqueness:
    @pytest.mark.parametrize(
        ('real_data', 'synthetic_data'),
        [
            [
                pd.Series(pd.Series([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])),
                pd.Series([1, 2, np.nan, 3, np.nan, 5, 2, np.nan, 6, None]),
            ],
            [
                pd.DataFrame({
                    'pk0': [f'id_{i}' for i in range(3)] * 2,
                    'pk1': [None] * 2 + [1] * 2 + [2] * 2,
                }),
                pd.DataFrame({
                    'id0': [None, 'id_0', 'id_1', 'id_0', 'id_0', 'id_1'],
                    'id1': [None, 0, 1, None, None, 1],
                }),
            ],
        ],
    )
    def test_compute_breakdown(self, real_data, synthetic_data):
        """Test the ``compute_breakdown`` method with Series and DataFrame input."""
        # Setup
        metric = KeyUniqueness()

        # Run
        result = metric.compute_breakdown(real_data, synthetic_data)

        # Assert
        assert result == {'score': 0.5}

    @pytest.mark.parametrize(
        ('real_data', 'synthetic_data'),
        [
            [
                pd.Series([1, 2, 3, 4, 5, 6, 7, 8, 9, 10] * 2),
                pd.Series([1, 2, np.nan, 3, np.nan, 5, 2, np.nan, 6, None]),
            ],
            [
                pd.DataFrame({
                    'pk0': [None, 'id0', 'id0', 'id1', 'id1'],
                    'pk1': [None, 0, 0, 1, 2],
                }),
                pd.DataFrame({
                    'id0': ['id0', 'id0', 'id1', 'id1', 'id1'],
                    'id1': [0, 1, 0, 1, 2],
                }),
            ],
        ],
    )
    @patch('sdmetrics.single_column.statistical.key_uniqueness.LOGGER')
    def test_compute_breakdown_with_duplicates_in_real_data(
        self, logger_mock, real_data, synthetic_data
    ):
        """Test the ``compute_breakdown`` method with duplicates in the real data."""
        # Setup
        metric = KeyUniqueness()

        # Run
        metric.compute_breakdown(real_data, synthetic_data)

        # Assert
        expected_message = 'The real data contains NA or duplicate values.'
        logger_mock.info.assert_called_once_with(expected_message)

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
        compute_breakdown_mock.assert_called_once_with(real_data, synthetic_data)
        assert result == 0.6
