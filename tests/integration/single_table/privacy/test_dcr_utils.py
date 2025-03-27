from datetime import datetime

import pandas as pd

from sdmetrics.single_table.privacy.dcr_utils import (
    calculate_dcr,
)


def test_calculate_dcr():
    """Test calculate_dcr with numerical values."""
    # Setup
    real_data_num = [0, 5, 8, 9, 10]
    synthetic_data_num_diff = [3, 5]

    real_df = pd.DataFrame({'num_col': real_data_num})
    synthetic_df_diff = pd.DataFrame({
        'num_col': synthetic_data_num_diff,
    })
    metadata = {'columns': {'num_col': {'sdtype': 'numerical'}}}

    # Run
    result = calculate_dcr(reference_dataset=real_df, dataset=synthetic_df_diff, metadata=metadata)

    # Assert
    expected_result = pd.Series([0.2, 0.0])
    pd.testing.assert_series_equal(result, expected_result)


def test_calculate_dcr_with_zero_col_range():
    """Test calculate_dcr with a range of zero."""
    # Setup
    real_data_num = [5.0]
    real_data_date = [datetime(2025, 1, 5)]
    synthetic_data_num_diff = [1, 2, 3, 5, 5]
    synthetic_data_date_diff = [
        datetime(2025, 1, 1),
        datetime(2025, 1, 2),
        datetime(2025, 1, 3),
        datetime(2025, 1, 4),
        datetime(2025, 1, 5),
    ]

    real_df = pd.DataFrame({'num_col': real_data_num, 'date_col': real_data_date})
    synthetic_df_diff = pd.DataFrame({
        'num_col': synthetic_data_num_diff,
        'date_col': synthetic_data_date_diff,
    })
    metadata = {'columns': {'num_col': {'sdtype': 'numerical'}, 'date_col': {'sdtype': 'datetime'}}}

    # Run
    result = calculate_dcr(reference_dataset=real_df, dataset=synthetic_df_diff, metadata=metadata)

    # Assert
    expected_result = pd.Series([1.0, 1.0, 1.0, 0.5, 0.0])
    pd.testing.assert_series_equal(result, expected_result)
