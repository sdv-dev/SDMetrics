import random
from datetime import datetime, timezone

import numpy as np
import pandas as pd
import pytest

from sdmetrics.single_table.privacy.dcr_utils import (
    _calculate_dcr_between_row_and_data,
    _calculate_dcr_between_rows,
    _calculate_dcr_value,
    _covert_datetime_cols_unix_timestamp,
    _to_unix_timestamp,
    calculate_dcr,
)
from tests.utils import check_if_value_in_threshold


@pytest.fixture()
def real_data():
    return pd.DataFrame({
        'num_col': [10, 20, np.nan, 40, 50, 60],
        'cat_col': ['A', 'B', 'A', None, 'B', 'C'],
        'bool_col': [True, False, True, False, None, False],
        'datetime_str_col': [
            '2025-01-01',
            '2025-01-31',
            '2025-01-10',
            '2025-01-21',
            None,
            '2025-01-12',
        ],
        'datetime_col': [
            datetime(2025, 1, 1),
            datetime(2025, 1, 31),
            datetime(2025, 1, 10),
            datetime(2025, 1, 21),
            pd.NaT,
            datetime(2025, 1, 12),
        ],
    })


@pytest.fixture()
def synthetic_data():
    return pd.DataFrame({
        'num_col': [10, 25, 30, 21, 7, np.nan],
        'cat_col': ['C', None, 'A', None, 'B', 'C'],
        'bool_col': [False, True, True, False, True, None],
        'datetime_str_col': [
            '2025-01-10',
            '2025-01-22',
            None,
            '2025-01-02',
            '2025-01-30',
            '2025-01-06',
        ],
        'datetime_col': [
            datetime(2025, 1, 10),
            datetime(2025, 1, 22),
            pd.NaT,
            datetime(2025, 1, 2),
            datetime(2025, 1, 30),
            datetime(2025, 1, 6),
        ],
    })


@pytest.fixture()
def column_ranges():
    return {
        'num_col': 50.0,
        'cat_col': 50.0,
        'bool_col': 50.0,
        'datetime_str_col': 30 * SECONDS_IN_DAY,
        'datetime_col': 30 * SECONDS_IN_DAY,
    }


@pytest.fixture()
def expected_row_comparisons():
    return [
        [0.0, 1.0, 1.0, 0.3, 0.3],
        [0.2, 1.0, 0.0, 0.7, 0.7],
        [1.0, 1.0, 1.0, 0.0, 0.0],
        [0.6, 1.0, 0.0, 0.366667, 0.366667],
        [0.8, 1.0, 1.0, 1.0, 1.0],
        [1.0, 0.0, 0.0, 0.066667, 0.066667],
        [0.3, 1.0, 0.0, 0.7, 0.7],
        [0.1, 1.0, 1.0, 0.3, 0.3],
        [1.0, 1.0, 0.0, 0.4, 0.4],
        [0.3, 0.0, 1.0, 0.033333, 0.033333],
        [0.5, 1.0, 1.0, 1.0, 1.0],
        [0.7, 1.0, 1.0, 0.333333, 0.333333],
        [0.4, 0.0, 0.0, 1.0, 1.0],
        [0.2, 1.0, 1.0, 1.0, 1.0],
        [1.0, 0.0, 0.0, 1.0, 1.0],
        [0.2, 1.0, 1.0, 1.0, 1.0],
        [0.4, 1.0, 1.0, 0.0, 0.0],
        [0.6, 1.0, 1.0, 1.0, 1.0],
        [0.22, 1.0, 1.0, 0.033333, 0.033333],
        [0.02, 1.0, 0.0, 0.966667, 0.966667],
        [1.0, 1.0, 1.0, 0.266667, 0.266667],
        [0.38, 0.0, 0.0, 0.633333, 0.633333],
        [0.58, 1.0, 1.0, 1.0, 1.0],
        [0.78, 1.0, 0.0, 0.333333, 0.333333],
        [0.06, 1.0, 0.0, 0.966667, 0.966667],
        [0.26, 0.0, 1.0, 0.033333, 0.033333],
        [1.0, 1.0, 0.0, 0.666667, 0.666667],
        [0.66, 1.0, 1.0, 0.3, 0.3],
        [0.86, 0.0, 1.0, 1.0, 1.0],
        [1.0, 1.0, 1.0, 0.6, 0.6],
        [1.0, 1.0, 1.0, 0.166667, 0.166667],
        [1.0, 1.0, 1.0, 0.833333, 0.833333],
        [0.0, 1.0, 1.0, 0.133333, 0.133333],
        [1.0, 1.0, 1.0, 0.5, 0.5],
        [1.0, 1.0, 0.0, 1.0, 1.0],
        [1.0, 0.0, 1.0, 0.2, 0.2],
    ]


@pytest.fixture()
def expected_same_dcr_result():
    return pd.Series([0.0, 0.0, 0.0, 0.0, 0.0, 0.0])


@pytest.fixture()
def expected_dcr_result():
    return pd.Series([0.226666, 0.273333, 0.48, 0.329333, 0.265333, 0.453333])


@pytest.fixture()
def test_metadata():
    return {
        'columns': {
            'num_col': {
                'sdtype': 'numerical',
            },
            'cat_col': {
                'sdtype': 'categorical',
            },
            'bool_col': {
                'sdtype': 'boolean',
            },
            'datetime_str_col': {
                'sdtype': 'datetime',
                'datetime_format': '%Y-%m-%d',
            },
            'datetime_col': {
                'sdtype': 'datetime',
            },
        },
    }


@pytest.fixture()
def converted_datetimes_data(real_data, synthetic_data):
    # Convert to timestamps, this conversion happens in
    # calculate_dcr so transforming now for test.
    def convert_to_timestamp(val):
        return val.timestamp() if pd.notna(val) else pd.NaT

    d_str_col = 'datetime_str_col'
    d_col = 'datetime_col'

    synthetic_data[d_str_col] = pd.to_datetime(synthetic_data[d_str_col], errors='coerce').apply(
        convert_to_timestamp
    )
    synthetic_data[d_col] = pd.to_datetime(synthetic_data[d_col], errors='coerce').apply(
        convert_to_timestamp
    )
    real_data[d_str_col] = pd.to_datetime(real_data[d_str_col], errors='coerce').apply(
        convert_to_timestamp
    )
    real_data[d_col] = pd.to_datetime(real_data[d_col], errors='coerce').apply(convert_to_timestamp)

    return real_data, synthetic_data


SECONDS_IN_DAY = 86400
ACCURACY_THRESHOLD = 0.000001


@pytest.mark.parametrize(
    's_value, d_value, col_range, sdtype, expected_dist',
    [
        (2.0, 2.0, 10.0, 'numerical', 0.0),
        (1.0, 2.0, 10.0, 'numerical', 0.1),
        (2.0, 1.0, 10.0, 'numerical', 0.1),
        (100.0, 1.0, 10.0, 'numerical', 1.0),
        (None, 1.0, 10.0, 'numerical', 1.0),
        (1.0, np.nan, 10.0, 'numerical', 1.0),
        (np.nan, None, 10.0, 'numerical', 0.0),
        ('A', 'B', None, 'categorical', 1.0),
        ('B', 'B', None, 'categorical', 0.0),
        ('B', 'A', None, 'categorical', 1.0),
        (None, 'B', None, 'categorical', 1.0),
        ('A', None, None, 'categorical', 1.0),
        (None, None, None, 'categorical', 0.0),
        (True, False, None, 'boolean', 1.0),
        (True, True, None, 'boolean', 0.0),
        (False, True, None, 'boolean', 1.0),
        (True, None, None, 'boolean', 1.0),
        (datetime(2025, 1, 1).timestamp(), None, SECONDS_IN_DAY, 'datetime', 1.0),
        (None, datetime(2025, 1, 1).timestamp(), SECONDS_IN_DAY, 'datetime', 1.0),
        (
            datetime(2025, 1, 1).timestamp(),
            datetime(2025, 1, 1).timestamp(),
            SECONDS_IN_DAY,
            'datetime',
            0.0,
        ),
        (
            datetime(2025, 1, 2).timestamp(),
            datetime(2025, 1, 1).timestamp(),
            2 * SECONDS_IN_DAY,
            'datetime',
            0.5,
        ),
        (
            datetime(2025, 10, 10).timestamp(),
            datetime(2025, 1, 1).timestamp(),
            SECONDS_IN_DAY,
            'datetime',
            1.0,
        ),
    ],
)
def test__calculate_dcr_value(s_value, d_value, col_range, sdtype, expected_dist):
    """Test _calculate_dcr_value with different types of values."""
    # Run
    dist = _calculate_dcr_value(s_value, d_value, sdtype, col_range)

    # Assert
    assert dist == expected_dist


def test__calculate_dcr_value_missing_range():
    """Test _calculate_dcr_value with missing range for numerical values."""
    # Setup
    error_message = (
        'No col_range was provided. The col_range is required '
        'for numerical and datetime sdtype DCR calculation.'
    )

    # Assert
    with pytest.raises(ValueError, match=error_message):
        _calculate_dcr_value(1, 1, 'numerical', None)


def test__calculate_dcr_missing_column(test_metadata):
    """Test _calculate_dcr_value with a missing column."""
    # Setup
    bad_col_name = 'col1'
    test_data = pd.DataFrame({bad_col_name: [1.0]})
    error_message = f'Column {bad_col_name} was not found in the metadata.'

    # Assert
    with pytest.raises(ValueError, match=error_message):
        calculate_dcr(test_data, test_data, test_metadata)


def test__calculate_dcr_between_rows(
    converted_datetimes_data, test_metadata, column_ranges, expected_row_comparisons
):
    """Test _calculate_dcr_between_rows for all row combinations"""
    # Setup
    result = []
    real_data, synthetic_data = converted_datetimes_data

    # Run
    for _, s_row_obj in synthetic_data.iterrows():
        for _, t_row_obj in real_data.iterrows():
            dist = _calculate_dcr_between_rows(s_row_obj, t_row_obj, column_ranges, test_metadata)
            result.append(dist)

    # Assert
    for i in range(len(expected_row_comparisons)):
        expected_dist = sum(expected_row_comparisons[i]) / len(expected_row_comparisons[i])
        check_if_value_in_threshold(result[i], expected_dist, ACCURACY_THRESHOLD)


def test__calculate_dcr_between_row_and_data(
    converted_datetimes_data, column_ranges, test_metadata, expected_dcr_result
):
    """Test _calculate_dcr_between_row_and_data for all rows."""
    # Setup
    result = []
    real_data, synthetic_data = converted_datetimes_data

    # Run
    for _, s_row_obj in synthetic_data.iterrows():
        dist = _calculate_dcr_between_row_and_data(
            s_row_obj, real_data, column_ranges, test_metadata
        )
        result.append(dist)

    # Assert
    for i in range(len(expected_dcr_result)):
        check_if_value_in_threshold(result[i], expected_dcr_result[i], ACCURACY_THRESHOLD)


def test_calculate_dcr(
    real_data,
    synthetic_data,
    test_metadata,
    expected_dcr_result,
    expected_same_dcr_result,
):
    """Calculate the DCR for all rows in a dataset against a traning dataset."""
    # Setup
    result_dcr = calculate_dcr(
        synthetic_data=synthetic_data, real_data=real_data, metadata=test_metadata
    )
    result_same_dcr = calculate_dcr(
        synthetic_data=synthetic_data, real_data=synthetic_data, metadata=test_metadata
    )

    # Assert
    pd.testing.assert_series_equal(result_dcr, expected_dcr_result)
    pd.testing.assert_series_equal(result_same_dcr, expected_same_dcr_result)


def test_calculate_dcr_missing_cols():
    """Test calculate_dcr with a missing column in synthetic data"""
    missing_col = 'missing_col'
    synthetic_df = pd.DataFrame({
        'col1': [0.0],
    })
    real_df = pd.DataFrame({'col1': [0.0], missing_col: [0.0]})
    metadata = {'columns': {'col1': {'sdtype': 'numerical'}, missing_col: {'sdtype': 'numerical'}}}

    error_message = "Different columns detected: {'missing_col'}"

    # Assert
    with pytest.raises(ValueError, match=error_message):
        calculate_dcr(synthetic_data=synthetic_df, real_data=real_df, metadata=metadata)


def test_calculate_dcr_bad_col(test_metadata):
    """Test calculate_dcr with a column not in metadata."""
    # Setup
    col_name = 'bad_col'
    fake_dataframe = pd.DataFrame({
        col_name: [0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
    })
    error_message = f'Column {col_name} was not found in the metadata.'

    # Assert
    with pytest.raises(ValueError, match=error_message):
        calculate_dcr(
            synthetic_data=fake_dataframe, real_data=fake_dataframe, metadata=test_metadata
        )


def test_calculate_dcr_with_shuffled_data():
    """Test calculate_dcr with even scores are unaffected by rows being shuffled."""
    # Setup
    real_data = [random.randint(1, 100) for _ in range(20)]
    synthetic_data = [random.randint(1, 100) for _ in range(20)]
    real_df = pd.DataFrame({'num_col': real_data})
    synthetic_df = pd.DataFrame({'num_col': synthetic_data})
    real_df_shuffled = pd.DataFrame({'num_col': random.sample(real_data, len(real_data))})
    synthetic_df_shuffled = pd.DataFrame({
        'num_col': random.sample(synthetic_data, len(synthetic_data))
    })

    metadata = {'columns': {'num_col': {'sdtype': 'numerical'}}}

    # Run
    result = calculate_dcr(synthetic_df, real_df, metadata)
    result_shuffled = calculate_dcr(synthetic_df_shuffled, real_df_shuffled, metadata)

    # Assert
    check_if_value_in_threshold(result.sum(), result_shuffled.sum(), 0.000001)


def test__to_unix_timestamp():
    # Setup
    not_datetime = 1
    actual_datetime = datetime(2025, 1, 1)
    timestamp = actual_datetime.timestamp()
    bad_type_msg = 'Value is not of type pandas datetime.'

    # Run
    with pytest.raises(ValueError, match=bad_type_msg):
        _to_unix_timestamp(not_datetime)

    with pytest.raises(ValueError, match=bad_type_msg):
        _to_unix_timestamp(timestamp)
    result = _to_unix_timestamp(actual_datetime)
    assert result == timestamp


def test__covert_datetime_cols_unix_timestamp():
    """Test _covert_datetime_cols_unix_timestamp to see if datetimes are converted."""
    # Setup
    int_cols = [1.0, 1.0, 2.0]
    datetime_cols = [
        datetime(2025, 1, 1, tzinfo=timezone.utc),
        datetime(2025, 1, 2, tzinfo=timezone.utc),
        datetime(2025, 1, 3, tzinfo=timezone.utc),
    ]
    timestamps = [val.timestamp() for val in datetime_cols]

    data = pd.DataFrame({
        'str_datetime_col': ['2025-01-01', '2025-01-02', '2025-01-03'],
        'datetime_col': datetime_cols,
        'int_col': int_cols,
        'str_col': ['2025-01-01', '2025-01-02', '2025-01-03'],
    })
    expected_data = pd.DataFrame({
        'str_datetime_col': timestamps,
        'datetime_col': datetime_cols,
        'int_col': int_cols,
        'str_col': ['2025-01-01', '2025-01-02', '2025-01-03'],
    })
    metadata = {
        'columns': {
            'str_datetime_col': {'sdtype': 'datetime', 'datetime_format': '%Y-%m-%d'},
            'datetime_col': {'sdtype': 'datetime', 'datetime_format': '%Y-%m-%d'},
            'int_col': {
                'sdtype': 'numerical',
            },
            'str_col': {
                'sdtype': 'str',
            },
        },
    }
    missing_format_metadata = {
        'columns': {
            'str_datetime_col': {'sdtype': 'datetime'},
            'datetime_col': {'sdtype': 'datetime'},
            'int_col': {
                'sdtype': 'numerical',
            },
            'str_col': {
                'sdtype': 'str',
            },
        },
    }

    # Run
    _covert_datetime_cols_unix_timestamp(data, metadata)

    # Assert
    pd.testing.assert_frame_equal(data, expected_data)
    with pytest.warns(UserWarning, match='No datetime format was specified.'):
        _covert_datetime_cols_unix_timestamp(data, missing_format_metadata)
