from datetime import datetime
import random
from unittest.mock import patch

import numpy as np
import pandas as pd
import pytest

from sdmetrics.single_table.privacy.dcr_baseline_protection import DCRBaselineProtection
from sdmetrics.utils import is_datetime


@pytest.fixture()
def test_data():
    real_data = pd.DataFrame({'num_col': [random.randint(1, 1000) for _ in range(50)]})
    synthetic_data = pd.DataFrame({'num_col': [random.randint(1, 1000) for _ in range(50)]})
    metadata = {'columns': {'num_col': {'sdtype': 'numerical'}}}
    return (real_data, synthetic_data, metadata)


class TestDCRBaselineProtection:
    @patch('sdmetrics.single_table.privacy.dcr_baseline_protection.DCRBaselineProtection._generate_random_data')
    @patch('sdmetrics.single_table.privacy.dcr_baseline_protection.calculate_dcr')
    def test_compute_breakdown(self, mock_calculate_dcr, mock_random_data, test_data):
        """Test that compute breakdown correctly measures the fraction of data overfitted."""
        # Setup
        real_data, synthetic_data, metadata = test_data
        num_iterations = 2
        num_rows_subsample = 5
        mock_value = 10.0
        mock_calculate_dcr_array = np.array([mock_value] * len(real_data))
        mock_calculate_dcr.return_value = pd.DataFrame(mock_calculate_dcr_array, columns=['dcr'])

        # Run
        result = DCRBaselineProtection.compute_breakdown(
            real_data, synthetic_data, metadata, num_rows_subsample, num_iterations
        )

        # Assert
        mock_random_data.assert_called_once()
        assert mock_calculate_dcr.call_count == 2 * num_iterations
        assert result['score'] == 1.0
        assert result['median_DCR_to_real_data']['synthetic_data'] == mock_value / num_iterations
        assert result['median_DCR_to_real_data']['random_data_baseline'] == mock_value / num_iterations

    @patch(
        'sdmetrics.single_table.privacy.dcr_overfitting_protection.DCROverfittingProtection.compute_breakdown'
    )
    def test_compute(self, mock_compute_breakdown, test_data):
        """Test that compute makes a call to compute_breakdown."""
        # Setup
        train_data, holdout_data, synthetic_data, metadata = test_data
        num_iterations = 2
        num_rows_subsample = 2

        # Run
        DCROverfittingProtection.compute(
            train_data, synthetic_data, holdout_data, metadata, num_rows_subsample, num_iterations
        )

        # Assert
        mock_compute_breakdown.assert_called_once_with(
            train_data, synthetic_data, holdout_data, metadata, num_rows_subsample, num_iterations
        )

    def test__generate_random_data_check_type_and_range(self):
        """Test generated random data contains the same value types and respect ranges."""
        # Setup
        real_data = pd.DataFrame({
            'num_col': [0, 0, np.nan, np.nan, 10, 10],
            'cat_col': ['A', 'B', 'A', None, 'B', None],
            'bool_col': [True, False, True, False, None, False],
            'unknown_column': [1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
            'datetime_col': [
                datetime(2025, 1, 1),
                datetime(2025, 1, 1),
                datetime(2025, 1, 11),
                datetime(2025, 1, 10),
                pd.NaT,
                datetime(2025, 1, 11),
            ],
        })

        # Run
        random_data = DCRBaselineProtection._generate_random_data(real_data)

        # Assert
        pd.testing.assert_series_equal(random_data.dtypes, real_data.dtypes)
        for col_name, col_data in random_data.items():
            if col_data.dtype in ['int64', 'int32', 'float64', 'float32'] or is_datetime(col_data):
                assert col_data.min() >= real_data[col_name].min()
                assert col_data.max() <= real_data[col_name].max()

    def test__generate_random_data_nan(self):
        """Test that nans are generated if the real data has nans."""
        # Setup
        real_data = pd.DataFrame({
            'float_col': np.repeat([1, 10, np.nan], 20),
            'cat_col': np.repeat(['A', 'B', None], 20),
        })

        # Run
        random_data = DCRBaselineProtection._generate_random_data(real_data)

        # Assert
        for col_name, col_data in random_data.items():
            if real_data[col_name].isna().any():
                assert col_data.isna().any()

    def test__generate_random_data_single_value(self):
        """Test that random generated data for a single value should be the original data."""
        # Setip
        real_data = pd.DataFrame({
            'float_col': [1.0],
        })

        # Run
        random_data = DCRBaselineProtection._generate_random_data(real_data)

        # Assert
        pd.testing.assert_frame_equal(real_data, random_data)

    def test_generate_random_data_different(self):
        """Test that generated data is differente everytime."""
        real_data = pd.DataFrame({'float_col': [1.0, 1000.0, 500.0], 'cat_col': ['A', 'B', 'C']})

        # Run
        random_data_1 = DCRBaselineProtection._generate_random_data(real_data)
        random_data_2 = DCRBaselineProtection._generate_random_data(real_data)

        # Assert
        with pytest.raises(AssertionError):
            pd.testing.assert_frame_equal(random_data_1, random_data_2)
