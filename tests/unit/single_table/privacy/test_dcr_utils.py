import re
from datetime import datetime

import numpy as np
import pandas as pd
import pytest

from sdmetrics.single_table.privacy.dcr_utils import (
    _calculate_dcr_dist,
    _calculate_dcr_dist_between_rows,
    _calculate_dist_between_row_and_data,
    calculate_dcr,
)


@pytest.fixture()
def train_data():
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
        'primary_key': 'student_id',
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


SECONDS_IN_DAY = 86400
ACCURACY_THRESHOLD = 0.000001


def check_if_value_in_threshold(value, expected_value, threshold=ACCURACY_THRESHOLD):
    assert abs(value - expected_value) < threshold


@pytest.mark.parametrize(
    's_value, d_value, range, col_name, expected_dist',
    [
        (2.0, 2.0, 10.0, 'num_col', 0.0),
        (1.0, 2.0, 10.0, 'num_col', 0.1),
        (2.0, 1.0, 10.0, 'num_col', 0.1),
        (100.0, 1.0, 10.0, 'num_col', 1.0),
        (None, 1.0, 10.0, 'num_col', 1.0),
        (1.0, np.nan, 10.0, 'num_col', 1.0),
        (np.nan, None, 10.0, 'num_col', 0.0),
        ('A', 'B', None, 'cat_col', 1.0),
        ('B', 'B', None, 'cat_col', 0.0),
        ('B', 'A', None, 'cat_col', 1.0),
        (None, 'B', None, 'cat_col', 1.0),
        ('A', None, None, 'cat_col', 1.0),
        (None, None, None, 'cat_col', 0.0),
        (True, False, None, 'bool_col', 1.0),
        (True, True, None, 'bool_col', 0.0),
        (False, True, None, 'bool_col', 1.0),
        (True, None, None, 'bool_col', 1.0),
        ('2025-01-10', None, SECONDS_IN_DAY, 'datetime_str_col', 1.0),
        (None, '2025-01-10', SECONDS_IN_DAY, 'datetime_str_col', 1.0),
        ('2025-01-10', '2025-01-10', SECONDS_IN_DAY, 'datetime_str_col', 0.0),
        ('2025-01-11', '2025-01-10', 2 * SECONDS_IN_DAY, 'datetime_str_col', 0.5),
        ('2025-02-11', '2025-01-10', SECONDS_IN_DAY, 'datetime_str_col', 1.0),
        (datetime(2025, 1, 1), None, SECONDS_IN_DAY, 'datetime_col', 1.0),
        (None, datetime(2025, 1, 1), SECONDS_IN_DAY, 'datetime_col', 1.0),
        (datetime(2025, 1, 1), datetime(2025, 1, 1), SECONDS_IN_DAY, 'datetime_col', 0.0),
        (datetime(2025, 1, 2), datetime(2025, 1, 1), 2 * SECONDS_IN_DAY, 'datetime_col', 0.5),
        (datetime(2025, 10, 10), datetime(2025, 1, 1), SECONDS_IN_DAY, 'datetime_col', 1.0),
    ],
)
def test__calculate_dcr_dist(s_value, d_value, range, col_name, expected_dist, test_metadata):
    """Test _calculate_dcr_dist with different types of values."""
    # Run
    dist = _calculate_dcr_dist(s_value, d_value, col_name, test_metadata, range)

    # Assert
    assert dist == expected_dist


def test__calculate_dcr_dist_missing_range(test_metadata):
    """Test _calculate_dcr_dist with missing range for numerical values."""
    # Setup
    col_name = 'num_col'
    error_message = (
        f'The numerical column: {col_name} did not produce a range. '
        'Check that column has sdtype=numerical and that it exists in training data.'
    )

    # Assert
    with pytest.raises(ValueError, match=error_message):
        _calculate_dcr_dist(1, 1, col_name, test_metadata, None)


def test__calculate_dcr_dist_missing_column(test_metadata):
    """Test _calculate_dcr_dist with a missing column."""
    # Setup
    col_name = 'bad_col'
    error_message = f'Column {col_name} was not found in the metadata.'

    # Assert
    with pytest.raises(ValueError, match=error_message):
        _calculate_dcr_dist(1, 1, col_name, test_metadata, 1)


def test__calculate_dcr_dist_between_rows(
    synthetic_data, train_data, test_metadata, column_ranges, expected_row_comparisons
):
    """Test _calculate_dcr_dist_between_rows for all row combinations"""
    # Setup
    result = []

    # Run
    for _, s_row_obj in synthetic_data.iterrows():
        for _, t_row_obj in train_data.iterrows():
            dist = _calculate_dcr_dist_between_rows(
                s_row_obj, t_row_obj, column_ranges, test_metadata
            )
            result.append(dist)

    # Assert
    for i in range(len(expected_row_comparisons)):
        expected_dist = sum(expected_row_comparisons[i]) / len(expected_row_comparisons[i])
        check_if_value_in_threshold(result[i], expected_dist)


def test__calculate_dcr_dist_between_rows_bad_index():
    """Run _calculate_dcr_dist_between_rows with an index that does not exist in comparison row."""
    # Setup
    synth_dataframe = pd.DataFrame({
        'A': [0.0, 0.0, 0.0],
        'B': [0.0, 0.0, 0.0],
    })
    train_dataframe = pd.DataFrame({
        'A': [0.0, 1.0, 1.0],
        'C': [1.0, 0.0, 1.0],
    })
    metadata = {'columns': {'A': {'sdtype': 'numerical'}, 'B': {'sdtype': 'numerical'}}}
    test_range = {'A': 1.0, 'C': 1.0}
    error_msg = re.escape('Column name (B) was not found when calculating DCR between two rows.')

    # Assert
    with pytest.raises(ValueError, match=error_msg):
        _calculate_dcr_dist_between_rows(
            synth_dataframe.iloc[0], train_dataframe.iloc[0], test_range, metadata
        )


def test__calculate_dist_between_row_and_data(
    synthetic_data, train_data, column_ranges, test_metadata, expected_dcr_result
):
    """Test _calculate_dist_between_row_and_data for all row combinations"""
    # Setup
    result = []

    # Run
    for _, s_row_obj in synthetic_data.iterrows():
        dist = _calculate_dist_between_row_and_data(
            s_row_obj, train_data, column_ranges, test_metadata
        )
        result.append(dist)

    # Assert
    for i in range(len(expected_dcr_result)):
        check_if_value_in_threshold(result[i], expected_dcr_result[i])


def test_calculate_dcr(
    train_data,
    synthetic_data,
    test_metadata,
    expected_dcr_result,
    expected_same_dcr_result,
):
    """Calculate the DCR for all rows in a dataset against a traning dataset."""
    # Setup
    result_dcr = calculate_dcr(
        synthetic_data=synthetic_data, comparison_data=train_data, metadata=test_metadata
    )
    result_same_dcr = calculate_dcr(
        synthetic_data=synthetic_data, comparison_data=synthetic_data, metadata=test_metadata
    )

    # Assert
    pd.testing.assert_series_equal(result_dcr, expected_dcr_result)
    pd.testing.assert_series_equal(result_same_dcr, expected_same_dcr_result)


def test_calculate_dcr_bad_col(test_metadata):
    """Test calculate_dcr with a missing column."""
    # Setup
    col_name = 'bad_col'
    fake_dataframe = pd.DataFrame({
        col_name: [0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
    })
    error_message = f'Column {col_name} was not found in the metadata.'

    # Assert
    with pytest.raises(ValueError, match=error_message):
        calculate_dcr(
            synthetic_data=fake_dataframe, comparison_data=fake_dataframe, metadata=test_metadata
        )
