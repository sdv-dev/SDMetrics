"""Utility Functions for Constraints."""

import re
from collections.abc import Iterable
from datetime import datetime

import numpy as np
import pandas as pd
from pandas.core.tools.datetimes import _guess_datetime_format_for_array

PRECISION_LEVELS = {
    '%Y': 1,  # Year
    '%y': 1,  # Year without century (same precision as %Y)
    '%B': 2,  # Full month name
    '%b': 2,  # Abbreviated month name
    '%m': 2,  # Month as a number
    '%d': 3,  # Day of the month
    '%j': 3,  # Day of the year
    '%U': 3,  # Week number (Sunday-starting)
    '%W': 3,  # Week number (Monday-starting)
    '%A': 3,  # Full weekday name
    '%a': 3,  # Abbreviated weekday name
    '%w': 3,  # Weekday as a decimal
    '%H': 4,  # Hour (24-hour clock)
    '%I': 4,  # Hour (12-hour clock)
    '%M': 5,  # Minute
    '%S': 6,  # Second
    '%f': 7,  # Microsecond
    # Formats that don't add precision
    '%p': 0,  # AM/PM
    '%z': 0,  # UTC offset
    '%Z': 0,  # Time zone name
    '%c': 0,  # Locale-based date/time
    '%x': 0,  # Locale-based date
    '%X': 0,  # Locale-based time
}


def _is_list_of_type(values, type_to_check=str):
    """Checks that 'values' is a list and all elements are of type 'type_to_check'."""
    return isinstance(values, list) and all(isinstance(value, type_to_check) for value in values)


def _tuple_from_columns(row, column_names):
    """Build a hashable tuple with the values that ``row`` has in ``column_names``.

    Every missing value is mapped to ``None`` so that two rows that are null in the
    same columns produce equal tuples.

    Args:
        row (pandas.Series):
            A pandas row.
        column_names (list[str]):
            The names of the columns to take the values from.

    Returns:
        tuple:
            The values of the row, in the order of ``column_names``.
    """
    return tuple(
        None if pd.isna(row[column_name]) else row[column_name] for column_name in column_names
    )


def _get_table_to_valid_rows(data):
    return {table: pd.Series(True, index=data[table].index) for table in data}


def _get_is_valid_dict(data, table_name):
    """Create a dictionary of True values for each table besides table_name.

    Besides table_name, all rows of every other table are considered valid,
    so the boolean Series will be True for all rows of every other table.

    Args:
        data (dict):
            The data.
        table_name (str):
            The name of the table to exclude from the dictionary.

    Returns:
        dict:
            Dictionary of table names to boolean Series of True values.
    """
    return {
        table: pd.Series(True, index=table_data.index)
        for table, table_data in data.items()
        if table != table_name or table_name is None
    }


def _cast_to_iterable(value, iterable_type=None):
    """Return a ``list`` if the input object is not a ``list`` or ``tuple``."""
    if isinstance(value, (list, tuple)):
        if iterable_type:
            return iterable_type(value)

        return value

    return [value]


def _get_datetime_format(value):
    """Get the ``strftime`` format for a given ``value``.

    This function returns the ``strftime`` format of a given ``value`` when possible.
    If the ``_guess_datetime_format_for_array`` from ``pandas.core.tools.datetimes`` is
    able to detect the ``strftime`` it will return it as a ``string`` if not, a ``None``
    will be returned.

    Args:
        value (pandas.Series, np.ndarray, list, or str):
            Input to attempt detecting the format.

    Return:
        String representing the datetime format in ``strftime`` format or ``None`` if not detected.
    """
    if not isinstance(value, pd.Series):
        value = pd.Series(value)

    value = value[~value.isna()]
    value = value.astype(str).to_numpy()

    return _guess_datetime_format_for_array(value)


def _is_datetime_type(value):
    """Determine if the input is a datetime type or not.

    If a ``pandas.Series`` or ``list`` is passed, it will return ``True`` if the first
    thousand values are datetime. Otherwise, it will check if the value is a datetime.

    Note: it will return ``False`` if ``value`` is a string representing
    a date before the year 1677.

    Args:
        value (array-like iterable, int, str or datetime):
            Input to evaluate.

    Returns:
        bool:
            True if the input is a datetime type, False if not.
    """
    if isinstance(value, str) or (not isinstance(value, Iterable)):
        value = _cast_to_iterable(value)

    values = pd.Series(value)
    values = values[~values.isna()]
    values = values.head(1000)  # only check 1000 values so this method takes less than 1 second
    for value in values:
        if not (
            bool(_get_datetime_format([value]))
            or isinstance(value, pd.Timestamp)
            or isinstance(value, datetime)
            or isinstance(value, pd.Period)
            or (isinstance(value, str) and pd.notna(pd.to_datetime(value, errors='coerce')))
        ):
            return False

    return True


def cast_to_datetime64(value, datetime_format=None, ignore_timezone=True):
    """Cast a given value to a ``numpy.datetime64`` format.

    Args:
        value (pandas.Series, np.ndarray, list, or str):
            Input data to convert.
        datetime_format (str, optional):
            Datetime format of the `value`.
        ignore_timezone (bool):
            If True, strips `%z` or `%Z` from the format and removes tzinfo.

    Returns:
        numpy.datetime64, pandas.Series, or numpy.ndarray of datetime64
    """
    if datetime_format:
        datetime_format = datetime_format.replace('%#', '%').replace('%-', '%')

    if isinstance(value, str):
        return _parse_datetime64_value(value, datetime_format, ignore_timezone)

    elif isinstance(value, pd.Series):
        dt_series = _parse_datetime(value, datetime_format, ignore_timezone)
        return dt_series.astype('datetime64[ns]')

    elif isinstance(value, (np.ndarray, list)):
        return np.array([
            _parse_datetime64_value(val, datetime_format, ignore_timezone) for val in value
        ])


def _parse_datetime64_value(value, datetime_format=None, ignore_timezone=True):
    """Parse a single value into `datetime64`, optionally ignoring timezone."""
    if pd.isna(value):
        return pd.NaT.to_datetime64()

    return _parse_datetime(value, datetime_format, ignore_timezone).to_datetime64()


def _parse_datetime(value, datetime_format, ignore_timezone):
    is_series = isinstance(value, pd.Series)
    parsed_value = pd.to_datetime(value, format=datetime_format, errors='coerce')

    if is_series and ignore_timezone and hasattr(parsed_value, 'dt'):
        if hasattr(parsed_value.dt, 'tz_localize'):
            parsed_value = parsed_value.dt.tz_localize(None)

    elif ignore_timezone and hasattr(parsed_value, 'tz_localize'):
        if isinstance(parsed_value, (list, tuple, pd.Series, np.ndarray)):
            parsed_value = [
                new_value.replace(tzinfo=None)
                if isinstance(new_value, datetime)
                else new_value.tz_localize(None)
                for new_value in parsed_value
            ]

        else:
            parsed_value = parsed_value.tz_localize(None)

    if is_series and not isinstance(parsed_value, pd.Series):
        return pd.Series(parsed_value)

    return parsed_value


def get_datetime_format_precision(format_str):
    """Return the precision level of a datetime format string."""
    # Find all format codes in the format string
    found_formats = re.findall(r'%[A-Za-z]', format_str)
    found_levels = (
        PRECISION_LEVELS.get(found_format)
        for found_format in found_formats
        if found_format in PRECISION_LEVELS
    )

    return max(found_levels, default=0)


def format_datetime_array(datetime_array, target_format):
    """Format each element in a numpy datetime64 array to a specified string format.

    Args:
        datetime_array (np.ndarray):
            Array of datetime64[ns] elements.
        target_format (str):
            The datetime format to cast each element to.

    Returns:
        np.ndarray: Array of formatted datetime strings.
    """
    return np.array([
        pd.to_datetime(date).strftime(target_format) if not pd.isna(date) else pd.NaT
        for date in datetime_array
    ])


def get_lower_precision_format(primary_format, secondary_format):
    """Compare two datetime format strings and return the one with lower precision.

    Args:
        primary_format (str):
            The first datetime format string to compare.
        low_precision_format (str):
            The second datetime format string to compare.

    Returns:
        str:
            The datetime format string with the lower precision level.
    """
    primary_level = get_datetime_format_precision(primary_format)
    secondary_level = get_datetime_format_precision(secondary_format)
    if primary_level >= secondary_level:
        return secondary_format

    return primary_format


def downcast_datetime_to_lower_precision(data, target_format):
    """Convert a datetime string from a higher-precision format to a lower-precision format.

    Args:
        data (np.array):
            The data to cast to the `target_format`.
        target_format (str):
            The datetime string to downcast.

    Returns:
        str: The datetime string in the lower precision format.
    """
    downcasted_data = format_datetime_array(data, target_format)
    return cast_to_datetime64(downcasted_data, target_format)


def match_datetime_precision(low, high, low_datetime_format, high_datetime_format):
    """Match `low` or `high` datetime array to the lower precision format.

    Args:
        low (np.ndarray):
            Array of datetime values for the low column.
        high (np.ndarray):
            Array of datetime values for the high column.
        low_datetime_format (str):
            The datetime format of the `low` column.
        high_datetime_format (str):
            The datetime format of the `high` column.

    Returns:
        Tuple[np.ndarray, np.ndarray]:
            Adjusted `low` and `high` arrays where the higher precision format is
            downcasted to the lower precision format.
    """
    lower_precision_format = get_lower_precision_format(low_datetime_format, high_datetime_format)
    if lower_precision_format == high_datetime_format:
        low = downcast_datetime_to_lower_precision(low, lower_precision_format)
    else:
        high = downcast_datetime_to_lower_precision(high, lower_precision_format)

    return low, high


def get_nan_component_value(row):
    """Check for NaNs in a pandas row.

    Outputs a concatenated string of the column names with NaNs.

    Args:
        row (pandas.Series):
            A pandas row.

    Returns:
        A concatenated string of the column names with NaNs.
    """
    columns_with_nans = []
    for column, value in row.items():
        if pd.isna(value):
            columns_with_nans.append(column)

    if columns_with_nans:
        return ', '.join(columns_with_nans)

    return 'None'


def compute_nans_column(table_data, list_column_names):
    """Compute a categorical column to the table_data indicating where NaNs are.

    Args:
        table_data (pandas.DataFrame):
            The table data.
        list_column_names (list):
            The list of column names to check for NaNs.

    Returns:
        A dict with the column name as key and the column indicating where NaNs are as value.
        Empty dict if there are no NaNs.
    """
    nan_column_name = '#'.join(list_column_names) + '.nan_component'
    column = table_data[list_column_names].apply(get_nan_component_value, axis=1)
    if not (column == 'None').all():
        return pd.Series(column, name=nan_column_name)

    return None
