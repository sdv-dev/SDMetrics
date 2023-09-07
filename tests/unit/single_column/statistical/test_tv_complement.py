from unittest.mock import patch

import numpy as np
import pandas as pd
import pytest

from sdmetrics.errors import IncomputableMetricError
from sdmetrics.single_column.statistical import TVComplement


class TestTVComplement:

    def test_compute(self):
        """Test the ``compute`` method.

        Expect that the total variation complement is returned.

        Input:
        - Real data.
        - Synthetic data.

        Output:
        - The evaluated metric.
        """
        # Setup
        real_data = pd.Series(['a', 'b', 'c', 'a', 'a', 'b'])
        synthetic_data = pd.Series(['a', 'b', 'c', 'a', 'b', 'c'])

        metric = TVComplement()

        # Run
        result = metric.compute(real_data, synthetic_data)

        # Assert
        assert result == 0.8333333333333333

    def test_compute_with_all_nans_synthetic_data(self):
        """Test it when the synthetic data is all NaNs."""
        # Setup
        real_data = pd.Series(['a', 'b', np.nan, 'a', np.nan, 'b'])
        synthetic_data = pd.Series([np.nan, np.nan, np.nan, np.nan, np.nan, np.nan])
        metric = TVComplement()

        # Run and Assert
        err_msg = 'The TVComplement metric must have 1 or more non-null values.'
        with pytest.raises(IncomputableMetricError, match=err_msg):
            metric.compute(real_data, synthetic_data)

    def test_compute_with_all_nans_real_data(self):
        """Test it when the real data is all NaNs."""
        # Setup
        real_data = pd.Series([np.nan, np.nan, np.nan, np.nan, np.nan, np.nan])
        synthetic_data = pd.Series(['a', 'b', np.nan, 'a', np.nan, 'b'])
        metric = TVComplement()

        # Run and Assert
        err_msg = 'The TVComplement metric must have 1 or more non-null values.'
        with pytest.raises(IncomputableMetricError, match=err_msg):
            metric.compute(real_data, synthetic_data)

    def test_compute_one_data_point(self):
        """Test it when data is mostly nans."""
        # Setup
        real_data = pd.Series([np.nan, np.nan, np.nan, np.nan, np.nan, 'b'])
        synthetic_data = pd.Series(['b', np.nan, np.nan, np.nan, np.nan, np.nan])
        metric = TVComplement()

        # Run
        result = metric.compute(real_data, synthetic_data)

        # Assert
        assert result == 1

    def test_compute_no_data(self):
        """Test it when no data is passed."""
        # Setup
        real_data = pd.Series()
        synthetic_data = pd.Series()
        metric = TVComplement()

        # Run and Assert
        err_msg = 'The TVComplement metric must have 1 or more non-null values.'
        with pytest.raises(IncomputableMetricError, match=err_msg):
            metric.compute(real_data, synthetic_data)

    @patch('sdmetrics.single_column.statistical.tv_complement.SingleColumnMetric.normalize')
    def test_normalize(self, normalize_mock):
        """Test the ``normalize`` method.

        Expect that the inherited ``normalize`` method is called.

        Input:
        - Raw score

        Output:
        - The output of the inherited ``normalize`` method.
        """
        # Setup
        metric = TVComplement()
        raw_score = 0.9

        # Run
        result = metric.normalize(raw_score)

        # Assert
        normalize_mock.assert_called_once_with(raw_score)
        assert result == normalize_mock.return_value
