from datetime import datetime
from unittest.mock import patch

import numpy as np
import pandas as pd

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
        real_data = pd.Series([
            datetime(2020, 10, 1),
            datetime(2021, 1, 2),
            datetime(2021, 9, 12),
            datetime(2022, 10, 1),

        ], dtype='datetime64[ns]')
        real_data_nans = pd.Series([
            datetime(2020, 10, 1),
            datetime(2021, 1, 2),
            datetime(2021, 9, 12),
            datetime(2022, 10, 1),
            pd.NaT
        ], dtype='datetime64[ns]')
        synthetic_data = pd.Series([
            datetime(2020, 11, 1),
            datetime(2021, 1, 2),
            datetime(2021, 2, 9),
            pd.NaT,
        ], dtype='datetime64[ns]')

        metric = BoundaryAdherence()

        # Run
        result = metric.compute(real_data, synthetic_data)
        result_ignore_nans = metric.compute(real_data_nans, synthetic_data)

        # Assert
        assert result == 0.75
        assert result_ignore_nans == 1

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
