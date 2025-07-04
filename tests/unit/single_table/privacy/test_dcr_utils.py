import random
from datetime import datetime

import numpy as np
import pandas as pd
import pytest

from sdmetrics.single_table.privacy.dcr_utils import (
    _process_dcr_chunk,
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
        dataset=synthetic_data, reference_dataset=real_data, metadata=test_metadata
    )
    result_same_dcr = calculate_dcr(
        dataset=synthetic_data, reference_dataset=synthetic_data, metadata=test_metadata
    )

    # Assert
    pd.testing.assert_series_equal(result_dcr, expected_dcr_result)
    pd.testing.assert_series_equal(result_same_dcr, expected_same_dcr_result)


def test_calculate_dcr_different_cols_in_metadata(real_data, synthetic_data, test_metadata):
    """Test that only intersecting columns of metadata, synthetic and real data are measured."""
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
        dataset=synthetic_data, reference_dataset=real_data, metadata=test_metadata
    )
    expected_result = pd.Series([0.0, 0.1, 0.2, 0.02, 0.06, 0.0])

    # Assert
    pd.testing.assert_series_equal(result, expected_result)

    test_metadata['columns'].pop('num_col')
    error_msg = 'There are no overlapping statistical columns to measure.'
    with pytest.raises(ValueError, match=error_msg):
        calculate_dcr(dataset=synthetic_data, reference_dataset=real_data, metadata=test_metadata)


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
    result = calculate_dcr(dataset=synthetic_df, reference_dataset=real_df, metadata=metadata)
    result_shuffled = calculate_dcr(
        dataset=synthetic_df_shuffled, reference_dataset=real_df_shuffled, metadata=metadata
    )

    # Assert
    check_if_value_in_threshold(result.sum(), result_shuffled.sum(), 0.000001)


@pytest.mark.filterwarnings('error')
def test_calculate_dcr_with_zero_range():
    """Test calculate_dcr with a range of zero."""
    # Setup
    real_data_num = [5.0]
    real_data_date = datetime(2025, 1, 1)
    synthetic_data_num_diff = [3.0]
    synthetic_data_date_diff = datetime(2025, 1, 2)
    synthetic_data_num_same = [5.0]
    synthetic_data_date_same = datetime(2025, 1, 1)
    real_df = pd.DataFrame({'num_col': real_data_num, 'date_col': real_data_date})
    synthetic_df_diff = pd.DataFrame({
        'num_col': synthetic_data_num_diff,
        'date_col': synthetic_data_date_diff,
    })
    synthetic_df_same = pd.DataFrame({
        'num_col': synthetic_data_num_same,
        'date_col': synthetic_data_date_same,
    })
    synthetic_df_half = pd.DataFrame({
        'num_col': synthetic_data_num_diff,
        'date_col': synthetic_data_date_same,
    })
    metadata = {'columns': {'num_col': {'sdtype': 'numerical'}, 'date_col': {'sdtype': 'datetime'}}}

    # Run and Assert
    result = calculate_dcr(reference_dataset=real_df, dataset=synthetic_df_diff, metadata=metadata)
    assert result[0] == 1.0
    result = calculate_dcr(reference_dataset=real_df, dataset=synthetic_df_same, metadata=metadata)
    assert result[0] == 0.0
    result = calculate_dcr(reference_dataset=real_df, dataset=synthetic_df_half, metadata=metadata)
    assert result[0] == 0.5


def test__process_dcr_chunk(real_data, synthetic_data, test_metadata, column_ranges):
    """Test '_process_dcr_chunk' to see if subset is properly dropped."""
    # Setup
    real_data = real_data.drop(columns=['datetime_str_col'])
    synthetic_data = synthetic_data.drop(columns=['datetime_str_col'])
    del test_metadata['columns']['datetime_str_col']
    cols_to_keep = real_data.columns

    synthetic_data['index'] = range(len(synthetic_data))
    real_data['index'] = range(len(real_data))
    rows_to_keep = 2
    chunk = synthetic_data.iloc[0:rows_to_keep].copy()

    # Run
    result = _process_dcr_chunk(
        dataset_chunk=chunk,
        reference_chunk=real_data,
        cols_to_keep=cols_to_keep,
        metadata=test_metadata,
        ranges=column_ranges,
    )

    # Assert
    assert isinstance(result, pd.Series)
    expected_result_first = 0.213333
    check_if_value_in_threshold(result[0], expected_result_first, 0.000001)
    assert len(result) == rows_to_keep
