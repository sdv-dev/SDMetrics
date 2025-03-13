import random
from datetime import datetime

import numpy as np
import pandas as pd
import pytest

from sdmetrics._utils_metadata import _convert_datetime_columns
from sdmetrics.single_table.privacy.dcr_utils import (
    _calculate_dcr_between_row_and_data,
    _calculate_dcr_between_rows,
    _calculate_dcr_value,
    calculate_dcr,
)
from tests.utils import check_if_value_in_threshold


@pytest.fixture()
def real_data():
    return pd.DataFrame({
        'num_col': [10, 20, np.nan, 40, 50, 60],
        'cat_col': ['A', 'B', 'A', None, 'B', 'C'],
        'cat_int_col': [1, 2, 1, None, 2, 3],
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
        'cat_int_col': [3, None, 1, None, 2, 3],
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
        'datetime_str_col': 30 * SECONDS_IN_DAY,
        'datetime_col': 30 * SECONDS_IN_DAY,
    }


@pytest.fixture()
def expected_row_comparisons():
    return [
        0.6,
        0.6,
        0.666666,
        0.555555,
        0.966666,
        0.188888,
        0.616666,
        0.616666,
        0.633333,
        0.227777,
        0.916666,
        0.727777,
        0.399999,
        0.866666,
        0.5,
        0.866667,
        0.566667,
        0.933333,
        0.547777,
        0.658888,
        0.755555,
        0.274444,
        0.93,
        0.574444,
        0.665555,
        0.221111,
        0.722222,
        0.71,
        0.643333,
        0.866666,
        0.7222222,
        0.944444,
        0.544444,
        0.833333,
        0.833333,
        0.4,
    ]


@pytest.fixture()
def expected_same_dcr_result():
    return pd.Series([0.0, 0.0, 0.0, 0.0, 0.0, 0.0])


@pytest.fixture()
def expected_dcr_result():
    return pd.Series([0.188888, 0.2277777, 0.4, 0.274444, 0.221111, 0.4])


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
            'cat_int_col': {
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
                'datetime_format': '%Y-%m-%d',
            },
        },
    }


SECONDS_IN_DAY = 86400
ACCURACY_THRESHOLD = 0.000001


@pytest.mark.parametrize(
    'synthetic_value, real_value, col_range, sdtype, expected_dist',
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
        (0, None, None, 'categorical', 1.0),
        (np.nan, None, None, 'categorical', 0.0),
        (np.nan, np.nan, None, 'categorical', 0.0),
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
def test__calculate_dcr_value(synthetic_value, real_value, col_range, sdtype, expected_dist):
    """Test _calculate_dcr_value with different types of values."""
    # Run
    dist = _calculate_dcr_value(synthetic_value, real_value, sdtype, col_range)

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


def test__calculate_dcr_between_rows(
    real_data, synthetic_data, test_metadata, column_ranges, expected_row_comparisons
):
    """Test _calculate_dcr_between_rows for all row combinations"""
    # Setup
    result = []
    real_data = _convert_datetime_columns(real_data, test_metadata)
    synthetic_data = _convert_datetime_columns(synthetic_data, test_metadata)

    # Run
    for _, s_row_obj in synthetic_data.iterrows():
        for _, t_row_obj in real_data.iterrows():
            dist = _calculate_dcr_between_rows(s_row_obj, t_row_obj, column_ranges, test_metadata)
            result.append(dist)

    # Assert
    for i in range(len(expected_row_comparisons)):
        check_if_value_in_threshold(result[i], expected_row_comparisons[i], ACCURACY_THRESHOLD)


def test__calculate_dcr_between_row_and_data(
    real_data, synthetic_data, column_ranges, test_metadata, expected_dcr_result
):
    """Test _calculate_dcr_between_row_and_data for all rows."""
    # Setup
    result = []
    real_data = _convert_datetime_columns(real_data, test_metadata)
    synthetic_data = _convert_datetime_columns(synthetic_data, test_metadata)

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
    # Run
    result_dcr = calculate_dcr(
        synthetic_data=synthetic_data, real_data=real_data, metadata=test_metadata
    )
    result_same_dcr = calculate_dcr(
        synthetic_data=synthetic_data, real_data=synthetic_data, metadata=test_metadata
    )

    # Assert
    pd.testing.assert_series_equal(result_dcr, expected_dcr_result)
    pd.testing.assert_series_equal(result_same_dcr, expected_same_dcr_result)


def test_calculate_dcr_different_cols_in_metadata(real_data, synthetic_data, test_metadata):
    """Test that only intersecting columns of metadata, synthetic data and real data are measured."""
    # Setup
    metadata_drop_columns = ['bool_col', 'datetime_col', 'cat_int_col', 'datetime_str_col']
    for col in metadata_drop_columns:
        test_metadata['columns'].pop(col)
    synthetic_data_drop_columns = ['cat_col', 'datetime_str_col']
    synthetic_data = synthetic_data.drop(columns=synthetic_data_drop_columns)
    real_data_drop_columns = ['bool_col', 'datetime_col']
    real_data = real_data.drop(columns=real_data_drop_columns)

    # Run
    result = calculate_dcr(
        synthetic_data=synthetic_data, real_data=real_data, metadata=test_metadata
    )
    expected_result = pd.Series([0.0, 0.1, 0.2, 0.02, 0.06, 0.0])

    # Assert
    pd.testing.assert_series_equal(result, expected_result)

    test_metadata['columns'].pop('num_col')
    error_msg = 'There are no overlapping statistical columns to measure.'
    with pytest.raises(ValueError, match=error_msg):
        calculate_dcr(synthetic_data=synthetic_data, real_data=real_data, metadata=test_metadata)


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
