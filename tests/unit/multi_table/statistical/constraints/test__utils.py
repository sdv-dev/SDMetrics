from datetime import datetime
from unittest.mock import patch

import numpy as np
import pandas as pd

from sdmetrics.multi_table.statistical.constraints._utils import (
    _create_unique_name,
    _get_datetime_format,
    _is_datetime_type,
    _is_list_of_type,
    _parse_datetime,
    _parse_datetime64_value,
    _replace_nans_with_none,
    cast_to_datetime64,
    compute_nans_column,
    downcast_datetime_to_lower_precision,
    format_datetime_array,
    get_datetime_format_precision,
    get_lower_precision_format,
    get_nan_component_value,
    match_datetime_precision,
)


def test__create_unique_name():
    """Test the ``_create_unique_name`` method."""
    # Setup
    name = 'name'
    existing_names = ['name', 'name_', 'name__']

    # Run
    result = _create_unique_name(name, existing_names)

    # Assert
    assert result == 'name___'


def test__replace_nans_with_none():
    """Test the `_replace_nans_with_none` method."""
    # Setup
    serie = pd.Series([1, 2, np.nan, 4, 5, np.nan, None])

    # Run
    result = _replace_nans_with_none(serie)

    # Assert
    expected = pd.Series([1, 2, None, 4, 5, None, None], dtype='object')
    pd.testing.assert_series_equal(result, expected)


def test__is_list_of_type():
    """Test `_is_list_of_type` method"""
    assert _is_list_of_type(['a', 'b'])
    assert not _is_list_of_type(['a', 1])
    assert not _is_list_of_type([1, 2])
    assert not _is_list_of_type(1)
    assert not _is_list_of_type('a')


def test__get_datetime_format():
    """Test the ``_get_datetime_format``.

    Setup:
        - string value representing datetime.
        - list of values with a datetime.
        - series with a datetime.

    Output:
        - The expected output is the format of the datetime representation.
    """
    # Setup
    string_value = '2021-02-02'
    list_value = [np.nan, '2021-02-02']
    series_value = pd.Series(['2021-02-02T12:10:59'])

    # Run
    string_out = _get_datetime_format(string_value)
    list_out = _get_datetime_format(list_value)
    series_out = _get_datetime_format(series_value)

    # Assert
    expected_output = '%Y-%m-%d'
    assert string_out == expected_output
    assert list_out == expected_output
    assert series_out == '%Y-%m-%dT%H:%M:%S'


def test__is_datetime_type_with_datetime_series():
    """Test the ``_is_datetime_type`` function when a datetime series is passed.

    Expect to return True when a datetime series is passed.

    Input:
    - A pandas.Series of type `datetime64[ns]`
    Output:
    - True
    """
    # Setup
    data = pd.Series(
        [pd.to_datetime('2020-01-01'), pd.to_datetime('2020-01-02'), pd.to_datetime('2020-01-03')],
    )

    # Run
    is_datetime = _is_datetime_type(data)

    # Assert
    assert is_datetime


def test__is_datetime_type_with_period():
    """Test the ``_is_datetime_type`` function when a period series is passed.

    Expect to return True when a period series is passed.

    Input:
    - A pandas.Series of type `period`
    Output:
    - True
    """
    # Setup
    data = pd.Series(pd.period_range('2023-01', periods=3, freq='M'))

    # Run
    is_datetime = _is_datetime_type(data)

    # Assert
    assert is_datetime


def test__is_datetime_type_with_mixed_array():
    """Test the ``_is_datetime_type`` function with a list of mixed datetime types."""
    # Setup
    data = [
        pd.to_datetime('2020-01-01'),
        '1890-03-05',
        pd.Timestamp('01-01-01'),
        datetime(2020, 1, 1),
        np.nan,
    ]

    # Run
    is_datetime = _is_datetime_type(data)

    # Assert
    assert is_datetime


def test__is_datetime_type_with_invalid_strings_in_list():
    """Test the ``_is_datetime_type`` function with a invalid datetime in a list."""
    # Setup
    data = [
        pd.to_datetime('2020-01-01'),
        '1890-03-05',
        pd.Timestamp('01-01-01'),
        datetime(2020, 1, 1),
        'invalid',
        np.nan,
    ]

    # Run
    is_datetime = _is_datetime_type(data)

    # Assert
    assert is_datetime is False


def test__is_datetime_type_with_datetime():
    """Test the ``_is_datetime_type`` function when a datetime is passed.

    Expect to return True when a datetime variable is passed.

    Input:
    - datetime.Datetime
    Output:
    - True
    """
    # Setup
    data = datetime(2020, 1, 1)

    # Run
    is_datetime = _is_datetime_type(data)

    # Assert
    assert is_datetime


def test__is_datetime_type_with_timestamp():
    """Test the ``_is_datetime_type`` function when a Timestamp is passed.

    Expect to return True when a datetime variable is passed.

    Input:
    - datetime.Datetime
    Output:
    - True
    """
    # Setup
    data = pd.Timestamp('2020-01-10')
    is_datetime = _is_datetime_type(data)

    # Assert
    assert is_datetime


def test__is_datetime_type_with_pandas_datetime():
    """Test the ``_is_datetime_type`` function when a pandas.datetime is passed.

    Expect to return True when a datetime variable is passed.

    Input:
    - pandas.Datetime
    Output:
    - True
    """
    # Setup
    data = pd.to_datetime('2020-01-01')

    # Run
    is_datetime = _is_datetime_type(data)

    # Assert
    assert is_datetime


def test__is_datetime_type_with_int():
    """Test the ``_is_datetime_type`` function when an int is passed.

    Expect to return False when an int variable is passed.

    Input:
    - int
    Output:
    - False
    """
    # Setup
    data = 2

    # Run
    is_datetime = _is_datetime_type(data)

    # Assert
    assert is_datetime is False


def test__is_datetime_type_with_datetime_str():
    """Test the ``_is_datetime_type`` function when an valid datetime string is passed.

    Expect to return True when a valid string representing datetime is passed.

    Input:
    - string
    Output:
    - True
    """
    # Setup
    value = '2021-02-02'

    # Run
    is_datetime = _is_datetime_type(value)

    # Assert
    assert is_datetime


def test__is_datetime_type_with_datetime_str_nanoseconds():
    """Test it for a datetime string with nanoseconds."""
    # Setup
    value = '2011-10-15 20:11:03.498707'

    # Run
    is_datetime = _is_datetime_type(value)

    # Assert
    assert is_datetime


def test__is_datetime_type_with_str_int():
    """Test it for a string with an integer."""
    # Setup
    value = '123'

    # Run
    is_datetime = _is_datetime_type(value)

    # Assert
    assert is_datetime is False


def test__is_datetime_type_with_invalid_str():
    """Test the ``_is_datetime_type`` function when an invalid string is passed.

    Expect to return False when an invalid string is passed.

    Input:
    - string
    Output:
    - False
    """
    # Setup
    value = 'abcd'

    # Run
    is_datetime = _is_datetime_type(value)

    # Assert
    assert is_datetime is False


def test__is_datetime_type_with_int_series():
    """Test the ``_is_datetime_type`` function when an int series is passed.

    Expect to return False when an int series variable is passed.

    Input:
    -  pd.Series of type int
    Output:
    - False
    """
    # Setup
    data = pd.Series([1, 2, 3, 4])

    # Run
    is_datetime = _is_datetime_type(data)

    # Assert
    assert is_datetime is False


def test_cast_to_datetime64():
    """Test the ``cast_to_datetime64`` function.

    Setup:
        - String value representing a datetime
        - List value with a ``np.nan`` and string values.
        - pd.Series with datetime values.
    Output:
        - A single np.datetime64
        - A list of np.datetime64
        - A series of np.datetime64
    """
    # Setup
    string_value = '2021-02-02'
    list_value = [None, np.nan, '2021-02-02']
    series_value = pd.Series(['2021-02-02', None, pd.NaT])

    # Run
    string_out = cast_to_datetime64(string_value)
    list_out = cast_to_datetime64(list_value)
    series_out = cast_to_datetime64(series_value)

    # Assert
    expected_string_output = np.datetime64('2021-02-02')
    expected_series_output = pd.Series([
        np.datetime64('2021-02-02'),
        np.datetime64('NaT'),
        np.datetime64('NaT'),
    ])
    expected_list_output = np.array(
        [np.datetime64('NaT'), np.datetime64('NaT'), '2021-02-02'], dtype='datetime64[ns]'
    )
    np.testing.assert_array_equal(expected_list_output, list_out)
    pd.testing.assert_series_equal(expected_series_output, series_out)
    assert expected_string_output == string_out


def test_cast_to_datetime64_datetime_format():
    """Test it when `datetime_format` is passed."""
    # Setup
    string_value = '2021-02-02'
    list_value = [None, np.nan, '2021-02-02']
    series_value = pd.Series(['2021-02-02', None, pd.NaT])

    # Run
    string_out = cast_to_datetime64(string_value, datetime_format='%Y-%m-%d')
    list_out = cast_to_datetime64(list_value, datetime_format='%Y-%m-%d')
    series_out = cast_to_datetime64(series_value, datetime_format='%Y-%m-%d')

    # Assert
    expected_string_output = np.datetime64('2021-02-02')
    expected_series_output = pd.Series([
        np.datetime64('2021-02-02'),
        np.datetime64('NaT'),
        np.datetime64('NaT'),
    ])
    expected_list_output = np.array(
        [np.datetime64('NaT'), np.datetime64('NaT'), '2021-02-02'], dtype='datetime64[ns]'
    )
    np.testing.assert_array_equal(expected_list_output, list_out)
    pd.testing.assert_series_equal(expected_series_output, series_out)
    assert expected_string_output == string_out


def test_cast_to_datetime64_ignore_timezone():
    """Test `cast_to_datetime64` with timezone-aware inputs and ignore_timezone=True."""
    # Setup
    string_value = '2021-02-02 10:00:00 -0500'
    list_value = [None, np.nan, '2021-02-02 10:00:00 -0500']
    series_value = pd.Series(['2021-02-02 10:00:00 -0500', None, pd.NaT])

    datetime_format = '%Y-%m-%d %H:%M:%S %z'

    # Run
    string_out = cast_to_datetime64(string_value, datetime_format=datetime_format)
    list_out = cast_to_datetime64(list_value, datetime_format=datetime_format)
    series_out = cast_to_datetime64(series_value, datetime_format=datetime_format)

    # Assert
    expected_string_output = np.datetime64('2021-02-02T10:00:00')
    expected_series_output = pd.Series([
        np.datetime64('2021-02-02T10:00:00'),
        np.datetime64('NaT'),
        np.datetime64('NaT'),
    ])
    expected_list_output = np.array(
        [np.datetime64('NaT'), np.datetime64('NaT'), np.datetime64('2021-02-02T10:00:00')],
        dtype='datetime64[ns]',
    )

    np.testing.assert_array_equal(expected_list_output, list_out)
    pd.testing.assert_series_equal(expected_series_output, series_out)
    assert expected_string_output == string_out


def test__parse_datetime64_value():
    """Test `_parse_datetime64_value` with valid date string and format."""
    # Setup
    value = '2021-02-02'
    expected = np.datetime64('2021-02-02')

    # Run
    result = _parse_datetime64_value(value, datetime_format='%Y-%m-%d')

    # Assert
    assert result == expected


def test__parse_datetime64_value_with_nat():
    """Test `_parse_datetime64_value` with NaN input returns NaT."""
    # Run
    result_none = _parse_datetime64_value(None)
    result_nan = _parse_datetime64_value(np.nan)

    # Assert
    assert np.isnat(result_none)
    assert np.isnat(result_nan)


def test__parse_datetime64_value_ignores_timezone():
    """Test `_parse_datetime64_value` strips timezone info when ignore_timezone=True."""
    # Setup
    value = '2021-02-02 15:00:00+0200'
    dt_format = '%Y-%m-%d %H:%M:%S%z'

    # Run
    result = _parse_datetime64_value(value, datetime_format=dt_format, ignore_timezone=True)

    # Assert
    assert isinstance(result, np.datetime64)
    assert str(result) == '2021-02-02T15:00:00.000000000'


def test__parse_datetime_with_series_and_timezone_and_ignore_tz():
    """Test `_parse_datetime` on a Series with timezone info."""
    # Setup
    series = pd.Series(['2020-01-01 10:00:00+0000', '2021-01-01 12:00:00+0200'])
    dt_format = '%Y-%m-%d %H:%M:%S%z'

    # Run
    result = _parse_datetime(series, datetime_format=dt_format, ignore_timezone=True)

    # Assert
    assert isinstance(result, pd.Series)
    assert result.dt.tz is None


def test__parse_datetime_without_ignoring_timezone():
    """Test `_parse_datetime` keeps tz-aware timestamps when ignore_timezone=False."""
    # Setup
    value = '2021-02-02 12:00:00+0200'
    dt_format = '%Y-%m-%d %H:%M:%S%z'

    # Run
    result = _parse_datetime(value, datetime_format=dt_format, ignore_timezone=False)

    # Assert
    assert result.tzinfo is not None
    assert str(result).endswith('+02:00')


def test_get_datetime_format_precision_seconds():
    """Test `get_datetime_format_precision` with second-level precision."""
    # Setup
    format_str = '%Y-%m-%d %H:%M:%S'
    expected_precision = 6

    # Run
    result = get_datetime_format_precision(format_str)

    # Assert
    assert result == expected_precision


def test_get_datetime_format_precision_microseconds():
    """Test `get_datetime_format_precision` with microsecond-level precision."""
    # Setup
    format_str = '%Y-%m-%d %H:%M:%S.%f'
    expected_precision = 7

    # Run
    result = get_datetime_format_precision(format_str)

    # Assert
    assert result == expected_precision


def test_get_datetime_format_precision_minutes():
    """Test `get_datetime_format_precision` with minute-level precision."""
    # Setup
    format_str = '%Y-%m-%d %H:%M'
    expected_precision = 5

    # Run
    result = get_datetime_format_precision(format_str)

    # Assert
    assert result == expected_precision


def test_get_datetime_format_precision_days():
    """Test `get_datetime_format_precision` with day-level precision."""
    # Setup
    format_str = '%Y-%m-%d'
    expected_precision = 3

    # Run
    result = get_datetime_format_precision(format_str)

    # Assert
    assert result == expected_precision


def test_get_datetime_format_precision_no_precision():
    """Test `get_datetime_format_precision` with no precision format."""
    # Setup
    format_str = '%Y'
    expected_precision = 1

    # Run
    result = get_datetime_format_precision(format_str)

    # Assert
    assert result == expected_precision


def test_get_datetime_format_precision_mixed_format_higher_precision():
    """Test `get_datetime_format_precision` with mixed higher-precision format."""
    # Setup
    format_str = '%Y-%m-%d %H:%M:%S.%f %z'
    expected_precision = 7

    # Run
    result = get_datetime_format_precision(format_str)

    # Assert
    assert result == expected_precision


def test_format_datetime_array_with_lower_precision_format():
    """Test `format_datetime_array` formatting datetime array to a lower-precision format."""
    # Setup
    datetime_array = np.array(
        ['2024-11-13 12:30:45.123456789', '2024-11-13 13:45:30.987654321'], dtype='datetime64[ns]'
    )
    target_format = '%Y-%m-%d %H:%M:%S'
    expected_result = np.array(['2024-11-13 12:30:45', '2024-11-13 13:45:30'], dtype='O')

    # Run
    result = format_datetime_array(datetime_array, target_format)

    # Assert
    np.testing.assert_array_equal(result, expected_result)


def test_get_lower_precision_format_with_different_precision():
    """Test `get_lower_precision_format` with different precision levels."""
    # Setup
    primary_format = '%Y-%m-%d %H:%M:%S'
    secondary_format = '%Y-%m-%d %H:%M:%S.%f'

    # Run
    result = get_lower_precision_format(primary_format, secondary_format)

    # Assert
    assert result == primary_format


def test_get_lower_precision_format_with_equal_precision():
    """Test `get_lower_precision_format` when both formats have the same precision."""
    # Setup
    primary_format = '%Y-%m-%d %H:%M:%S'
    secondary_format = '%Y-%m-%d %H:%M:%S'

    # Run
    result = get_lower_precision_format(primary_format, secondary_format)

    # Assert
    assert result == secondary_format == primary_format


def test_get_lower_precision_format_with_date_only():
    """Test `get_lower_precision_format` with date-only formats."""
    # Setup
    primary_format = '%Y-%m-%d'
    secondary_format = '%Y-%m'

    # Run
    result = get_lower_precision_format(primary_format, secondary_format)

    # Assert
    assert result == secondary_format


def test_get_lower_precision_format_with_week_and_day_formats():
    """Test `get_lower_precision_format` with week and day level formats."""
    # Setup
    primary_format = '%Y-%W'
    secondary_format = '%Y-%m-%d'

    # Run
    result = get_lower_precision_format(primary_format, secondary_format)

    # Assert
    assert result == secondary_format


def test_downcast_datetime_to_lower_precision():
    """Test `downcast_datetime_to_lower_precision` to ensure datetime downcasting."""
    # Setup
    data = np.array(
        ['2024-11-13 12:30:45.123456789', '2024-11-13 13:45:30.987654321'], dtype='datetime64[ns]'
    )
    target_format = '%Y-%m-%d %H:%M:%S'
    expected_result = np.array(['2024-11-13 12:30:45', '2024-11-13 13:45:30'], dtype='O')

    # Run
    result = downcast_datetime_to_lower_precision(data, target_format)

    # Assert
    np.testing.assert_array_equal(result, cast_to_datetime64(expected_result))


def test_downcast_datetime_to_lower_precision_to_day():
    """Test `downcast_datetime_to_lower_precision` to downcast datetime to day precision."""
    # Setup
    data = np.array(
        ['2024-11-13 12:30:45.123456789', '2024-11-14 13:45:30.987654321'], dtype='datetime64[ns]'
    )
    target_format = '%Y-%m-%d'  # Downcasting to day precision
    expected_result = np.array(['2024-11-13', '2024-11-14'], dtype='O')

    # Run
    result = downcast_datetime_to_lower_precision(data, target_format)

    # Assert
    np.testing.assert_array_equal(result, cast_to_datetime64(expected_result))


@patch('sdmetrics.multi_table.statistical.constraints._utils.downcast_datetime_to_lower_precision')
def test_match_datetime_precision_low_has_higher_precision(mock_downcast):
    """Test `match_datetime_precision` when `low` has higher precision than `high`.

    This test checks that if the `low` array has a more precise format than `high`,
    `low` is downcasted to match the `high` format.
    """
    # Setup
    low = np.array(['2024-11-13 10:34:45.123456', '2024-11-14 12:20:10.654321'], dtype='O')
    high = np.array(['2024-11-13 10:34:45', '2024-11-14 12:20:10'], dtype='O')
    low_format = '%Y-%m-%d %H:%M:%S.%f'
    high_format = '%Y-%m-%d %H:%M:%S'
    expected_low = np.array(['2024-11-13 10:34:45', '2024-11-14 12:20:10'], dtype='O')

    # Set the return value of the mock to simulate downcasting
    mock_downcast.return_value = expected_low

    # Run
    result_low, result_high = match_datetime_precision(low, high, low_format, high_format)

    # Assert
    mock_downcast.assert_called_once_with(low, high_format)
    np.testing.assert_array_equal(result_low, expected_low)
    np.testing.assert_array_equal(result_high, high)


@patch('sdmetrics.multi_table.statistical.constraints._utils.downcast_datetime_to_lower_precision')
def test_match_datetime_precision_high_has_higher_precision(mock_downcast):
    """Test `match_datetime_precision` when `high` has higher precision than `low`.

    This test checks that if the `high` array has a more precise format than `low`,
    `high` is downcasted to match the `low` format.
    """
    # Setup
    low = np.array(['2024-11-13 10:34:45', '2024-11-14 12:20:10'], dtype='O')
    high = np.array(['2024-11-13 10:34:45.123456', '2024-11-14 12:20:10.654321'], dtype='O')
    low_format = '%Y-%m-%d %H:%M:%S'
    high_format = '%Y-%m-%d %H:%M:%S.%f'
    expected_high = np.array(['2024-11-13 10:34:45', '2024-11-14 12:20:10'], dtype='O')

    # Set the return value of the mock to simulate downcasting
    mock_downcast.return_value = expected_high

    # Run
    result_low, result_high = match_datetime_precision(low, high, low_format, high_format)

    # Assert
    mock_downcast.assert_called_once_with(high, low_format)
    np.testing.assert_array_equal(result_low, low)
    np.testing.assert_array_equal(result_high, expected_high)


def test_get_nan_component_value():
    """Test the ``get_nan_component_value`` method."""
    # Setup
    row = pd.Series([np.nan, 2, np.nan, 4], index=['a', 'b', 'c', 'd'])

    # Run
    result = get_nan_component_value(row)

    # Assert
    assert result == 'a, c'


def test_compute_nans_columns():
    """Test the ``compute_nans_columns`` method."""
    # Setup
    data = pd.DataFrame({
        'a': [1, np.nan, 3, np.nan],
        'b': [np.nan, 2, 3, np.nan],
        'c': [1, np.nan, 3, np.nan],
    })

    # Run
    output = compute_nans_column(data, ['a', 'b', 'c'])
    expected_output = pd.Series(['b', 'a, c', 'None', 'a, b, c'], name='a#b#c.nan_component')

    # Assert
    pd.testing.assert_series_equal(output, expected_output)


def test_compute_nans_columns_without_nan():
    """Test the ``compute_nans_columns`` method when there are no nans."""
    # Setup
    data = pd.DataFrame({'a': [1, 2, 3, 2], 'b': [2.5, 2, 3, 2.5], 'c': [1, 2, 3, 2]})

    # Run
    output = compute_nans_column(data, ['a', 'b', 'c'])

    # Assert
    assert output is None
