import re
from copy import deepcopy
from unittest.mock import patch

import pandas as pd
import pytest

from sdmetrics._utils_metadata import (
    _convert_datetime_column,
    _convert_datetime_columns,
    _process_data_with_metadata,
    _remove_missing_columns_metadata,
    _remove_non_modelable_columns,
    _validate_metadata,
    _validate_metadata_dict,
    _validate_multi_table_metadata,
    _validate_single_table_metadata,
)


@pytest.fixture
def data():
    return {
        'table1': pd.DataFrame({
            'numerical': [1, 2, 3],
            'categorical': ['a', 'b', 'c'],
            'datetime_str': ['2021-01-01', '2021-01-02', '2021-01-03'],
            'datetime': pd.to_datetime(['2025-01-01', '2025-01-02', '2025-01-03']),
        }),
        'table2': pd.DataFrame({
            'datetime_missing_format': ['2024-01-01', '2023-01-02', '2024-01-03'],
            'extra_column_1': [1, 2, 3],
            'extra_column_2': ['a', 'b', 'c'],
        }),
    }


@pytest.fixture
def metadata():
    return {
        'tables': {
            'table1': {
                'columns': {
                    'numerical': {'sdtype': 'numerical'},
                    'categorical': {'sdtype': 'categorical'},
                    'datetime_str': {'sdtype': 'datetime', 'datetime_format': '%Y-%m-%d'},
                    'datetime': {'sdtype': 'datetime'},
                }
            },
            'table2': {
                'columns': {
                    'datetime_missing_format': {'sdtype': 'datetime'},
                }
            },
        }
    }


def test__validate_metadata_dict(metadata):
    """Test the ``_validate_metadata_dict`` method."""
    # Setup
    metadata_wrong = 'wrong'
    expected_error = re.escape(
        f"Expected a dictionary but received a '{type(metadata_wrong).__name__}' instead."
        " For SDV metadata objects, please use the 'to_dict' function to convert it"
        ' to a dictionary.'
    )

    # Run and Assert
    _validate_metadata_dict(metadata)
    with pytest.raises(TypeError, match=expected_error):
        _validate_metadata_dict(metadata_wrong)


def test__validate_single_table_metadata(metadata):
    """Test the ``_validate_single_table_metadata`` method."""
    # Setup
    metadata_wrong = {
        'wrong_key': {
            'numerical': {'sdtype': 'numerical'},
            'categorical': {'sdtype': 'categorical'},
        }
    }
    expected_error = re.escape(
        "Single-table metadata must include a 'columns' key that maps column names"
        ' to their corresponding information.'
    )

    # Run and Assert
    _validate_single_table_metadata(metadata['tables']['table1'])
    with pytest.raises(ValueError, match=expected_error):
        _validate_single_table_metadata(metadata_wrong)


def test__validate_multi_table_metadata(metadata):
    """Test the ``_validate_multi_table_metadata`` method."""
    # Setup
    metadata_wrong = {
        'wrong_tables': {
            'table1': {
                'columns': {
                    'numerical': {'sdtype': 'numerical'},
                    'categorical': {'sdtype': 'categorical'},
                }
            },
        }
    }

    metadata_wrong_single_table = {
        'tables': {
            'table1': {
                'columns': {
                    'numerical': {'sdtype': 'numerical'},
                    'categorical': {'sdtype': 'categorical'},
                }
            },
            'table2': {
                'wrong_key': {
                    'numerical': {'sdtype': 'numerical'},
                    'categorical': {'sdtype': 'categorical'},
                }
            },
        }
    }
    expected_error = re.escape(
        "Multi-table metadata must include a 'tables' key that maps table names to"
        ' their respective metadata.'
    )
    expected_error_single_table = re.escape(
        "Error in table 'table2': Single-table metadata must include a 'columns' key"
        ' that maps column names to their corresponding information.'
    )

    # Run and Assert
    _validate_multi_table_metadata(metadata)
    with pytest.raises(ValueError, match=expected_error):
        _validate_multi_table_metadata(metadata_wrong)

    with pytest.raises(ValueError, match=expected_error_single_table):
        _validate_multi_table_metadata(metadata_wrong_single_table)


@patch('sdmetrics._utils_metadata._validate_multi_table_metadata')
def test__validate_metadata(mock_validate_multi_table_metadata, metadata):
    """Test the ``_validate_metadata`` method."""
    # Setup
    wrong_metadata = {'worng_key': 'wrong_value'}
    expected_error = re.escape(
        "Metadata must include either a 'columns' key for single-table metadata"
        " or a 'tables' key for multi-table metadata."
    )
    # Run
    _validate_metadata(metadata)
    with pytest.raises(ValueError, match=expected_error):
        _validate_metadata(wrong_metadata)

    # Assert
    mock_validate_multi_table_metadata.assert_called_once_with(metadata)


def test__convert_datetime_column(data, metadata):
    """Test the ``_convert_datetime_column`` method."""
    # Setup
    column_metadata = {'sdtype': 'datetime', 'datetime_format': '%Y-%m-%d'}
    str_col = pd.Series(['2021-01-01', '2021-01-02', '2021-01-03'])
    datetime = pd.Series([
        pd.Timestamp('2021-01-01'),
        pd.Timestamp('2021-01-02'),
        pd.Timestamp('2021-01-03'),
    ])

    # Run
    expected_msg = re.escape(
        "Datetime column 'datetime_no_format' does not have a specified 'datetime_format'. "
        'Please add a the required datetime_format to the metadata or convert this column '
        "to 'pd.datetime' to bypass this requirement."
    )
    with pytest.raises(ValueError, match=expected_msg):
        _convert_datetime_column('datetime_no_format', str_col, {'sdtype': 'datetime'})

    datetime_result = _convert_datetime_column('datetime', datetime, column_metadata)
    str_result = _convert_datetime_column('datetime_str', str_col, column_metadata)

    # Assert
    pd.testing.assert_series_equal(datetime, datetime_result)
    pd.testing.assert_series_equal(datetime, str_result)


def test__convert_datetime_column_bad_format(data, metadata):
    """Test the ``_convert_datetime_columns`` method when the provided format fails."""
    # Setup
    column_metadata = {'sdtype': 'datetime', 'datetime_format': '%Y-%m-%d'}
    bad_col = pd.Series(['bad', 'datetime', 'values'])

    # Run and assert
    expected_msg = re.escape("Error converting column 'datetime' to timestamp: ")
    with pytest.raises(ValueError, match=expected_msg):
        _convert_datetime_column('datetime', bad_col, column_metadata)


def test__convert_datetime_columns(data, metadata):
    """Test the ``_convert_datetime_columns`` method."""
    # Setup
    expected_df_single_table = pd.DataFrame({
        'numerical': [1, 2, 3],
        'categorical': ['a', 'b', 'c'],
        'datetime_str': pd.to_datetime(['2021-01-01', '2021-01-02', '2021-01-03']),
        'datetime': pd.to_datetime(['2025-01-01', '2025-01-02', '2025-01-03']),
    })
    expected_result_multi_table_table = {
        'table1': expected_df_single_table,
        'table2': data['table2'],
    }

    # Run
    error_msg = (
        "Datetime column 'datetime_missing_format' does not have a specified 'datetime_format'. "
        'Please add a the required datetime_format to the metadata or convert this column '
        "to 'pd.datetime' to bypass this requirement."
    )
    with pytest.raises(ValueError, match=error_msg):
        _convert_datetime_columns(data, metadata)

    table2_columns = metadata['tables']['table2']['columns']
    table2_columns['datetime_missing_format']['datetime_format'] = '%Y-%m-%d'
    result_multi_table = _convert_datetime_columns(data, metadata)
    result_single_table = _convert_datetime_columns(data['table1'], metadata['tables']['table1'])

    # Assert
    for table_name, table in result_multi_table.items():
        pd.testing.assert_frame_equal(table, expected_result_multi_table_table[table_name])

    pd.testing.assert_frame_equal(result_single_table, expected_df_single_table)


def test_convert_datetime_columns_with_failures():
    """Test the ``_convert_datetime_columns`` when pandas can't convert to datetime."""
    # Setup
    data = pd.DataFrame({
        'numerical': [1, 2, 3],
        'categorical': ['a', 'b', 'c'],
        'datetime_1': ['2021-01-01', '20-error', '2021-01-03'],
        'datetime_2': ['2025-01-01', '2025-01-24', '2025-01-04'],
    })

    metadata = {
        'columns': {
            'numerical': {'sdtype': 'numerical'},
            'categorical': {'sdtype': 'categorical'},
            'datetime_1': {'sdtype': 'datetime', 'datetime_format': '%Y-%m-%d'},
            'datetime_2': {'sdtype': 'datetime'},
        }
    }

    # Run and Assert
    error_msg_bad_format = 'match format'
    with pytest.raises(ValueError, match=error_msg_bad_format):
        _convert_datetime_columns(data, metadata)

    data['datetime_1'] = ['2021-01-01', '2021-01-02', '2021-01-03']

    error_msg_missing_format = "does not have a specified 'datetime_format'"
    with pytest.raises(ValueError, match=error_msg_missing_format):
        _convert_datetime_columns(data, metadata)

    metadata['columns']['datetime_2']['datetime_format'] = '%Y-%m-%d'

    result = _convert_datetime_columns(data, metadata)
    assert pd.api.types.is_datetime64_any_dtype(result['datetime_1'].dtype)
    assert pd.api.types.is_datetime64_any_dtype(result['datetime_2'].dtype)


def test__remove_missing_columns_metadata(data, metadata):
    """Test the ``_remove_missing_columns_metadata`` method."""
    # Setup
    expected_warning_missing_column_metadata = re.escape(
        "The columns ('extra_column_1', 'extra_column_2') are not present in the metadata. "
        'They will not be included for further evaluation.'
    )
    expected_warning_extra_metadata_column = re.escape(
        "The columns ('numerical') are in the metadata but they are not present in the data."
    )
    data['table1'] = data['table1'].drop(columns=['numerical'])

    # Run
    with pytest.warns(UserWarning, match=expected_warning_extra_metadata_column):
        _remove_missing_columns_metadata(data['table1'], metadata['tables']['table1'])

    with pytest.warns(UserWarning, match=expected_warning_missing_column_metadata):
        result = _remove_missing_columns_metadata(data['table2'], metadata['tables']['table2'])

    # Assert
    pd.testing.assert_frame_equal(
        result, data['table2'].drop(columns=['extra_column_1', 'extra_column_2'])
    )


def test__remove_missing_columns_metadata_with_single_table(data, metadata):
    """Test the ``_remove_missing_columns_metadata`` method with a single table."""
    # Setup
    expected_df_single_table = data['table2'].drop(columns=['extra_column_1', 'extra_column_2'])
    expected_df_multi_table = {
        'table1': data['table1'],
        'table2': expected_df_single_table,
    }

    # Run
    result_single_table = _remove_missing_columns_metadata(
        data['table2'], metadata['tables']['table2']
    )
    result_multi_table = _remove_missing_columns_metadata(data, metadata)

    # Assert
    pd.testing.assert_frame_equal(result_single_table, expected_df_single_table)
    for table_name, table in result_multi_table.items():
        pd.testing.assert_frame_equal(table, expected_df_multi_table[table_name])


def test__remove_non_modelable_columns(data, metadata):
    """Test the ``_remove_non_modelable_columns`` method."""
    # Setup
    single_table_df = pd.DataFrame({
        'numerical': [1, 2, 3],
        'categorical': ['a', 'b', 'c'],
        'id': [1, 2, 3],
        'boolean': [True, False, True],
        'datetime': pd.to_datetime(['2025-01-01', '2025-01-02', '2025-01-03']),
        'ssn': ['123-45-6789', '987-65-4321', '123-45-6789'],
    })
    metadata_single_table = {
        'columns': {
            'numerical': {'sdtype': 'numerical'},
            'categorical': {'sdtype': 'categorical'},
            'id': {'sdtype': 'id'},
            'boolean': {'sdtype': 'boolean'},
            'datetime': {'sdtype': 'datetime'},
            'ssn': {'sdtype': 'ssn'},
        }
    }
    multi_table = {
        'table1': deepcopy(single_table_df),
        'table2': data['table2'],
    }
    multi_table_metadata = {
        'tables': {
            'table1': metadata_single_table,
            'table2': metadata['tables']['table2'],
        }
    }

    # Run
    result_single_table = _remove_non_modelable_columns(single_table_df, metadata_single_table)
    result_multi_table = _remove_non_modelable_columns(multi_table, multi_table_metadata)

    # Assert
    pd.testing.assert_frame_equal(result_single_table, single_table_df.drop(columns=['id', 'ssn']))
    pd.testing.assert_frame_equal(
        result_multi_table['table1'], single_table_df.drop(columns=['id', 'ssn'])
    )


@patch('sdmetrics._utils_metadata._validate_metadata_dict')
@patch('sdmetrics._utils_metadata._remove_missing_columns_metadata')
@patch('sdmetrics._utils_metadata._convert_datetime_columns')
@patch('sdmetrics._utils_metadata._remove_non_modelable_columns')
def test__process_data_with_metadata(
    mock_remove_non_modelable_columns,
    mock_convert_datetime_columns,
    mock_remove_missing_columns_metadata,
    mock_validate_metadata_dict,
    data,
    metadata,
):
    """Test the ``_process_data_with_metadata``method."""
    # Setup
    mock_convert_datetime_columns.side_effect = lambda data, metadata: data
    mock_remove_missing_columns_metadata.side_effect = lambda data, metadata: data

    # Run and Assert
    _process_data_with_metadata(data, metadata)

    mock_convert_datetime_columns.assert_called_once_with(data, metadata)
    mock_remove_missing_columns_metadata.assert_called_once_with(data, metadata)
    mock_validate_metadata_dict.assert_called_once_with(metadata)
    mock_remove_non_modelable_columns.assert_not_called()

    _process_data_with_metadata(data, metadata, keep_modelable_columns_only=True)
    mock_remove_non_modelable_columns.assert_called_once_with(data, metadata)
