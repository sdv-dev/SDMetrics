from unittest.mock import patch

import numpy as np
import pandas as pd

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

    @patch('sdmetrics.single_column.statistical.category_adherence.'
           'CategoryAdherence.compute_breakdown')
    def test_compute(self, compute_breakdown_mock):
        """Test the ``compute`` method."""
        # Setup
        real_data = pd.Series(['A', 'B', 'C', 'B', 'A'])
        synthetic_data = pd.Series(['A', 'B', 'C', 'D', 'E'])
        metric = CategoryAdherence()
        compute_breakdown_mock.return_value = {'score': 0.6}

        # Run
        result = metric.compute(real_data, synthetic_data)

        # Assert
        compute_breakdown_mock.assert_called_once_with(real_data, synthetic_data)
        assert result == 0.6
