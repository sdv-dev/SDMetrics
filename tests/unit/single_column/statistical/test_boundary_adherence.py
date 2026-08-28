from datetime import datetime
from unittest.mock import patch

import numpy as np
import pandas as pd
import pytest

from sdmetrics.single_column.statistical import BoundaryAdherence


class TestBoundaryAdherence:
    def test_compute(self):
        """Test the ``compute`` method.

        Expect that the number of in-bounds values in the synthetic data is returned.

        Input:
        - Real data.
        - Synthetic data.

        Output:
        - The evaluated metric.
        """
        # Setup
        real_data = pd.Series([1.0, 2.4, 2.6, 0.8])
        synthetic_data = pd.Series([0.9, 1.8, 2.1, 5.0])

        metric = BoundaryAdherence()

        # Run
        result = metric.compute(real_data, synthetic_data)

        # Assert
        assert result == 0.75

    def test_compute_nans(self):
        """Test the ``compute`` method with nan values.

        Expect that the nan values in synthetic data are considered as
        out of bounds if the real data does not also containt nan values.
        """
        # Setup
        real_data = pd.Series([1.0, 2.4, 2.6, 0.8])
        real_data_nans = pd.Series([1.0, 2.4, 2.6, 0.8, np.nan])
        synthetic_data = pd.Series([0.9, 1.8, 2.1, 5.0, np.nan])

        metric = BoundaryAdherence()

        # Run
        result = metric.compute(real_data, synthetic_data)
        result_ignore_nans = metric.compute(real_data_nans, synthetic_data)

        # Assert
        assert result == 0.6
        assert result_ignore_nans == 0.75

    def test_compute_datetime_nans(self):
        """Test the ``compute`` method with nan values.

        Expect that the nan values in synthetic data are considered as
        out of bounds if the real data does not also containt nan values.
        """
        # Setup
        real_data = pd.Series(
            [
                datetime(2020, 10, 1),
                datetime(2021, 1, 2),
                datetime(2021, 9, 12),
                datetime(2022, 10, 1),
            ],
            dtype='datetime64[ns]',
        )
        real_data_nans = pd.Series(
            [
                datetime(2020, 10, 1),
                datetime(2021, 1, 2),
                datetime(2021, 9, 12),
                datetime(2022, 10, 1),
                pd.NaT,
            ],
            dtype='datetime64[ns]',
        )
        synthetic_data = pd.Series(
            [
                datetime(2020, 11, 1),
                datetime(2021, 1, 2),
                datetime(2021, 2, 9),
                pd.NaT,
            ],
            dtype='datetime64[ns]',
        )

        metric = BoundaryAdherence()

        # Run
        result = metric.compute(real_data, synthetic_data)
        result_ignore_nans = metric.compute(real_data_nans, synthetic_data)

        # Assert
        assert result == 0.75
        assert result_ignore_nans == 1

    @pytest.mark.parametrize(
        ('range_min', 'range_max', 'expected_score'),
        [
            (None, None, 0.75),
            (0, 10, 1.0),
            (0, None, 0.75),
            (1.0, None, 0.5),
            (None, 2.0, 0.5),
        ],
    )
    def test_compute_with_range_min_and_range_max(self, range_min, range_max, expected_score):
        """Test the ``compute`` method with ``range_min`` and ``range_max``.

        Expect that the given bounds are used instead of the ones of the real data,
        including when they are falsy values such as ``0``.
        """
        # Setup
        real_data = pd.Series([1.0, 2.4, 2.6, 0.8])
        synthetic_data = pd.Series([0.9, 1.8, 2.1, 5.0])

        metric = BoundaryAdherence()

        # Run
        result = metric.compute(real_data, synthetic_data, range_min=range_min, range_max=range_max)

        # Assert
        assert result == expected_score

    @pytest.mark.parametrize(
        ('real_data', 'range_is_nullable', 'expected_score'),
        [
            (pd.Series([1.0, 2.4, 2.6, 0.8]), None, 0.6),
            (pd.Series([1.0, 2.4, 2.6, 0.8, None]), None, 0.75),
            (pd.Series([1.0, 2.4, 2.6, 0.8]), True, 0.75),
            (pd.Series([1.0, 2.4, 2.6, 0.8, np.nan]), False, 0.6),
        ],
    )
    def test_compute_with_range_is_nullable(self, real_data, range_is_nullable, expected_score):
        """Test the ``compute`` method with ``range_is_nullable``.

        Expect that the nan values of the synthetic data are ignored if the column is
        nullable, and considered out of bounds otherwise. If ``range_is_nullable`` is not
        provided, it is determined by the real data.
        """
        # Setup
        synthetic_data = pd.Series([0.9, 1.8, np.nan, 2.1, 5.0])

        metric = BoundaryAdherence()

        # Run
        result = metric.compute(real_data, synthetic_data, range_is_nullable=range_is_nullable)

        # Assert
        assert result == expected_score

    @pytest.mark.parametrize(
        ('range_min', 'range_max', 'expected_score'),
        [
            (None, None, 1 / 3),
            ('2018-01-01', '2026-01-01', 1.0),
            (datetime(2018, 1, 1), datetime(2021, 1, 1), 2 / 3),
            (pd.Timestamp('2018-01-01'), pd.Timestamp('2026-01-01'), 1.0),
            (pd.Timestamp('2018-01-01'), '2026-01-01', 1.0),
            ('2018-01-01', datetime(2021, 1, 1), 2 / 3),
        ],
    )
    def test_compute_datetime_with_range_min_and_range_max(
        self, range_min, range_max, expected_score
    ):
        """Test the ``compute`` method with datetime bounds.

        Expect that the bounds can be passed as strings, datetimes or timestamps.
        """
        # Setup
        real_data = pd.Series(
            [datetime(2020, 10, 1), datetime(2021, 1, 2)],
            dtype='datetime64[ns]',
        )
        synthetic_data = pd.Series(
            [datetime(2019, 1, 1), datetime(2020, 11, 1), datetime(2025, 1, 1)],
            dtype='datetime64[ns]',
        )

        metric = BoundaryAdherence()

        # Run
        result = metric.compute(real_data, synthetic_data, range_min=range_min, range_max=range_max)

        # Assert
        assert result == expected_score

    def test_compute_breakdown_with_range(self):
        """Test that ``compute_breakdown`` passes the range information to ``compute``."""
        # Setup
        real_data = pd.Series([1.0, 2.4, 2.6, 0.8])
        synthetic_data = pd.Series([0.9, 1.8, 2.1, 5.0])

        metric = BoundaryAdherence()

        # Run
        result = metric.compute_breakdown(
            real_data, synthetic_data, range_min=0, range_max=10, range_is_nullable=True
        )

        # Assert
        assert result == {'score': 1.0}

    @patch('sdmetrics.single_column.statistical.boundary_adherence.SingleColumnMetric.normalize')
    def test_normalize(self, normalize_mock):
        """Test the ``normalize`` method.

        Expect that the inherited ``normalize`` method is called.

        Input:
        - Raw score

        Output:
        - The output of the inherited ``normalize`` method.
        """
        # Setup
        metric = BoundaryAdherence()
        raw_score = 0.9

        # Run
        result = metric.normalize(raw_score)

        # Assert
        normalize_mock.assert_called_once_with(raw_score)
        assert result == normalize_mock.return_value
