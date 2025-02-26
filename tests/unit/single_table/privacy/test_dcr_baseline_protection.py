from datetime import datetime
import pytest
import numpy as np
import pandas as pd
from sdmetrics.single_table.privacy.dcr_baseline_protection import DCRBaselineProtection
from sdmetrics.utils import is_datetime


class TestDCRBaselineProtection:
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
