import random
import re
from datetime import datetime
from unittest.mock import patch

import numpy as np
import pandas as pd
import pytest

from sdmetrics.single_table.privacy import DCRBaselineProtection
from sdmetrics.utils import is_datetime


@pytest.fixture()
def test_data():
    real_data = pd.DataFrame({'num_col': [random.randint(1, 1000) for _ in range(50)]})
    synthetic_data = pd.DataFrame({'num_col': [random.randint(1, 1000) for _ in range(50)]})
    metadata = {'columns': {'num_col': {'sdtype': 'numerical'}}}
    return (real_data, synthetic_data, metadata)


class TestDCRBaselineProtection:
    def test__validate_inputs(self, test_data):
        """Test that we properly validate inputs to our DCRBaselineProtection."""
        # Setup
        real_data, synthetic_data, metadata = test_data

        # Run and Assert
        zero_subsample_msg = re.escape('num_rows_subsample (0) must be an integer greater than 1.')
        with pytest.raises(ValueError, match=zero_subsample_msg):
            DCRBaselineProtection.compute_breakdown(real_data, synthetic_data, metadata, 0)

        subsample_none_msg = re.escape(
            'num_iterations should not be greater than 1 if there is no subsampling.'
        )
        with pytest.raises(ValueError, match=subsample_none_msg):
            DCRBaselineProtection.compute_breakdown(real_data, synthetic_data, metadata, None, 10)

        large_subsample_msg = re.escape('Ignoring the num_rows_subsample and num_iterations args.')
        with pytest.warns(UserWarning, match=large_subsample_msg):
            DCRBaselineProtection.compute_breakdown(
                real_data, synthetic_data, metadata, len(synthetic_data) * 2
            )

        zero_iteration_msg = re.escape('num_iterations (0) must be an integer greater than 1.')
        with pytest.raises(ValueError, match=zero_iteration_msg):
            DCRBaselineProtection.compute_breakdown(real_data, synthetic_data, metadata, 1, 0)

        no_dcr_metadata = {'columns': {'bad_col': {'sdtype': 'unknown'}}}
        no_dcr_data = pd.DataFrame({'bad_col': [1.0]})

        missing_metric = 'There are no overlapping statistical columns to measure.'
        with pytest.raises(ValueError, match=missing_metric):
            DCRBaselineProtection.compute_breakdown(no_dcr_data, no_dcr_data, no_dcr_metadata)

    @patch(
        'sdmetrics.single_table.privacy.dcr_baseline_protection.DCRBaselineProtection._generate_random_data'
    )
    @patch('sdmetrics.single_table.privacy.dcr_baseline_protection.calculate_dcr')
    def test_compute_breakdown(self, mock_calculate_dcr, mock_random_data, test_data):
        """Test that compute breakdown correctly measures the fraction of data overfitted."""
        # Setup
        real_data, synthetic_data, metadata = test_data
        num_iterations = 2
        num_rows_subsample = 5
        mock_value = 10.0
        mock_calculate_dcr_array = np.array([mock_value] * len(real_data))
        mock_calculate_dcr.return_value = pd.Series(mock_calculate_dcr_array)

        # Run
        result = DCRBaselineProtection.compute_breakdown(
            real_data, synthetic_data, metadata, num_rows_subsample, num_iterations
        )

        # Assert
        mock_random_data.assert_called_once()
        assert mock_calculate_dcr.call_count == 2 * num_iterations
        assert result['score'] == 1.0
        assert result['median_DCR_to_real_data']['synthetic_data'] == mock_value
        assert result['median_DCR_to_real_data']['random_data_baseline'] == mock_value

    @patch(
        'sdmetrics.single_table.privacy.dcr_baseline_protection.DCRBaselineProtection.compute_breakdown'
    )
    def test_compute(self, mock_compute_breakdown, test_data):
        """Test that compute makes a call to compute_breakdown."""
        # Setup
        real_data, synthetic_data, metadata = test_data
        num_iterations = 2
        num_rows_subsample = 2

        # Run
        DCRBaselineProtection.compute(
            real_data, synthetic_data, metadata, num_rows_subsample, num_iterations
        )

        # Assert
        mock_compute_breakdown.assert_called_once_with(
            real_data, synthetic_data, metadata, num_rows_subsample, num_iterations
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
            if pd.api.types.is_numeric_dtype(col_data) or is_datetime(col_data):
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
        # Setup
        real_data = pd.DataFrame({
            'float_col': [1.0],
        })

        # Run
        random_data = DCRBaselineProtection._generate_random_data(real_data)

        # Assert
        pd.testing.assert_frame_equal(real_data, random_data)

    def test_generate_random_data_different(self):
        """Test that generated data is different everytime."""
        # Setup
        real_data = pd.DataFrame({'float_col': [1.0, 1000.0, 500.0], 'cat_col': ['A', 'B', 'C']})

        # Run
        random_data_1 = DCRBaselineProtection._generate_random_data(real_data)
        random_data_2 = DCRBaselineProtection._generate_random_data(real_data)

        # Assert
        with pytest.raises(AssertionError):
            pd.testing.assert_frame_equal(random_data_1, random_data_2)

    @patch('sdmetrics.single_table.privacy.dcr_baseline_protection.calculate_dcr')
    def test_compute_breakdown_with_dcr_random_median_zero(self, mock_calculate_dcr, test_data):
        """Test compute_breakdown when random median dcr score is 0."""
        # Setup
        real_data, synthetic_data, metadata = test_data
        num_iterations = 2
        num_rows_subsample = 5
        mock_value = 0.0
        mock_calculate_dcr_array = np.array([mock_value] * len(real_data))
        mock_calculate_dcr.return_value = pd.Series(mock_calculate_dcr_array)

        # Run
        result = DCRBaselineProtection.compute_breakdown(
            real_data, synthetic_data, metadata, num_rows_subsample, num_iterations
        )

        # Assert
        assert result['median_DCR_to_real_data']['random_data_baseline'] == 0.0
        assert np.isnan(result['score'])

    @patch(
        'sdmetrics.single_table.privacy.dcr_baseline_protection.DCRBaselineProtection._generate_random_data'
    )
    def test_compute_breakdown_with_dcr_random_same_real(self, mock_generate_random, test_data):
        """Test compute breakdown if random data is the same as real data."""
        # Setup
        real_data, synthetic_data, metadata = test_data
        num_rows_subsample = 10
        mock_generate_random.return_value = real_data.copy()

        # Run
        result = DCRBaselineProtection.compute_breakdown(
            real_data, synthetic_data, metadata, num_rows_subsample
        )

        # Assert
        assert result['median_DCR_to_real_data']['random_data_baseline'] == 0.0
        assert np.isnan(result['score'])
        args = mock_generate_random.call_args[0]
        assert args[1] == len(synthetic_data)
        pd.testing.assert_frame_equal(args[0], real_data)
