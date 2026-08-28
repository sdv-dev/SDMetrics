from unittest.mock import patch

import numpy as np
import pandas as pd
import pytest

from sdmetrics.single_column.statistical import CategoryAdherence


class TestCategoryAdherence:
    def test_compute_breakdown(self):
        """Test the ``compute_breakdown`` method."""
        # Setup
        real_data = pd.Series(['A', 'B', 'C', 'B', 'A'])
        synthetic_data = pd.Series(['A', 'B', 'C', 'D', 'E'])

        metric = CategoryAdherence()

        # Run
        result = metric.compute_breakdown(real_data, synthetic_data)

        # Assert
        assert result == {'score': 0.6}

    def test_compute_breakdown_with_nans(self):
        """Test the ``compute_breakdown`` method with NaNs."""
        # Setup
        real_data = pd.Series(['A', 'B', 'C', 'B', 'A', None])
        synthetic_data = pd.Series(['A', 'B', np.nan, 'C', np.nan, 'B', 'A', None, 'D', 'C'])

        metric = CategoryAdherence()

        # Run
        result = metric.compute_breakdown(real_data, synthetic_data)

        # Assert
        assert result == {'score': 0.9}

    @pytest.mark.parametrize(
        ('real_data', 'range_values', 'range_is_nullable', 'expected_score'),
        [
            # ``range_values`` replaces the categories of the real data
            (pd.Series(['A', 'B']), ['A', 'B', 'C', 'D'], None, 0.75),
            (pd.Series(['A', 'B']), ['A', 'B', 'C'], None, 0.5),
            # nans are valid if the real data or the ``range_values`` contain them
            (pd.Series(['A', 'B', None]), ['A', 'B', 'C'], None, 0.75),
            (pd.Series(['A', 'B', 'D']), ['A', 'B', None], None, 0.75),
            (pd.Series(['A', 'B', 'D']), None, True, 1.0),
            # ``range_is_nullable`` takes precedence over the real data
            (pd.Series(['A', 'B', None, 'D']), None, False, 0.75),
            (pd.Series(['A', 'B', None]), ['A', 'B', 'C'], False, 0.5),
        ],
    )
    def test_compute_breakdown_with_range(
        self, real_data, range_values, range_is_nullable, expected_score
    ):
        """Test the ``compute_breakdown`` method with ``range_values`` and ``range_is_nullable``.

        Expect that the given range information is used instead of the one computed from the
        real data, and that any value that is not provided is computed from the real data.
        """
        # Setup
        synthetic_data = pd.Series(['A', 'B', np.nan, 'D'])

        metric = CategoryAdherence()

        # Run
        result = metric.compute_breakdown(
            real_data,
            synthetic_data,
            range_values=range_values,
            range_is_nullable=range_is_nullable,
        )

        # Assert
        assert result == {'score': expected_score}

    @pytest.mark.parametrize(
        ('range_values', 'range_is_nullable'),
        [
            (None, None),
            (['A', 'B', 'C', 'D', 'E'], False),
            (['A', 'B', 'C', 'D', 'E'], True),
        ],
    )
    @patch(
        'sdmetrics.single_column.statistical.category_adherence.CategoryAdherence.compute_breakdown'
    )
    def test_compute(self, compute_breakdown_mock, range_values, range_is_nullable):
        """Test the ``compute`` method passes the range information to ``compute_breakdown``."""
        # Setup
        real_data = pd.Series(['A', 'B', 'C', 'B', 'A'])
        synthetic_data = pd.Series(['A', 'B', 'C', 'D', 'E'])
        metric = CategoryAdherence()
        compute_breakdown_mock.return_value = {'score': 0.6}

        # Run
        result = metric.compute(
            real_data,
            synthetic_data,
            range_values=range_values,
            range_is_nullable=range_is_nullable,
        )

        # Assert
        compute_breakdown_mock.assert_called_once_with(
            real_data,
            synthetic_data,
            range_values=range_values,
            range_is_nullable=range_is_nullable,
        )
        assert result == 0.6
