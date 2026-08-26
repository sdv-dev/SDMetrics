"""Utility Functions for Constraints."""

import re
from collections.abc import Iterable
from datetime import datetime

import numpy as np
import pandas as pd
from pandas.core.tools.datetimes import _guess_datetime_format_for_array

from sdmetrics.multi_table.statistical.constraints.error import (
    ConstraintNotApplicableError,
)

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


class CustomNan:
    """Custom NaN class."""

    def __eq__(self, other):
        """Check if the other object is a CustomNan."""
        return isinstance(other, CustomNan)

    def __hash__(self):
        """Return a hash for the CustomNan."""
        return hash('CustomNaN')

    def __lt__(self, value):
        """Return that this class is always less than value."""
        return False

    def __gt__(self, value):
        """Return that this class is always greater than value."""
        return True

    def __str__(self):
        """Return a human-readable string representation of the object."""
        return 'CustomNan'

    def __repr__(self):
        """Return a human-readable string representation of the object."""
        return 'CustomNan()'


def _create_unique_name(name, list_names):
    """Modify the ``name`` parameter if it already exists in the list of names."""
    result = name
    while result in list_names:
        result += '_'

    return result


def _replace_nans_with_none(series):
    """Replace all NaN values in a pandas Series with None.

    Args:
        series (pd.Series): The input pandas Series.

    Returns:
        pd.Series: A new Series with NaN values replaced by None.
    """
    return series.astype(object).where(series.notna(), None)


def _tuple_from_columns(row, columns):
    """Return a tuple of values for ``columns`` using ``None`` in place of NaN."""
    return tuple(row[c] if not pd.isna(row[c]) else None for c in columns)


def _is_list_of_type(values, type_to_check=str):
    """Checks that 'values' is a list and all elements are of type 'type_to_check'."""
    return isinstance(values, list) and all(isinstance(value, type_to_check) for value in values)


def _get_key_values(table_data, key_columns):
    """Return one hashable value per row for the given key columns.

    Every missing value is mapped to ``None`` so that two rows that are null in the
    same columns produce equal values.

    Args:
        table_data (pandas.DataFrame):
            The data of the table.
        key_columns (list[str]):
            The names of the columns that make up the key.

    Returns:
        pandas.Series:
            A tuple with the values of ``key_columns`` for every row.
    """
    return pd.Series(
        [_tuple_from_columns(row, key_columns) for _, row in table_data[key_columns].iterrows()],
        index=table_data.index,
        dtype=object,
    )


def _validate_foreign_to_primary_key_subset_input(
    parent_table_name,
    child_table_name,
    child_foreign_key,
    conditional_column_name,
    conditional_values,
):
    """Validate the input for the ForeignToPrimaryKeySubset constraint."""
    if not isinstance(parent_table_name, str):
        raise TypeError('`parent_table_name` must be a string.')

    if not isinstance(child_table_name, str):
        raise TypeError('`child_table_name` must be a string.')

    if not isinstance(child_foreign_key, str) and not _is_list_of_type(child_foreign_key):
        raise TypeError('`child_foreign_key` must be a string or a list of strings.')

    if not isinstance(conditional_column_name, str):
        raise TypeError('`conditional_column_name` must be a string.')

    if not isinstance(conditional_values, list):
        raise TypeError('`conditional_values` must be a list.')


def _validate_foreign_to_primary_key_subset(
    data,
    parent_primary_key,
    parent_table_name,
    child_table_name,
    child_foreign_key,
    conditional_column_name,
    conditional_values,
):
    """Validate the ForeignToPrimaryKeySubset constraint."""
    parent_primary_key = _cast_to_iterable(parent_primary_key)
    child_foreign_key = _cast_to_iterable(child_foreign_key)
    indicator_col = _create_unique_name('_merge', parent_primary_key + child_foreign_key)
    parent_table = data[parent_table_name]
    merged_parent = (
        parent_table[parent_primary_key]
        .merge(
            data[child_table_name][child_foreign_key].drop_duplicates(),
            left_on=parent_primary_key,
            right_on=child_foreign_key,
            how='left',
            indicator=indicator_col,
        )
        .set_index(parent_table.index)
    )
    filtered_parent = parent_table[merged_parent[indicator_col] == 'both']
    table_to_valid_rows = _get_table_to_valid_rows(data)
    if not set(filtered_parent[conditional_column_name]).issubset(conditional_values):
        good_parent_value_index = filtered_parent[conditional_column_name].isin(conditional_values)
        good_parent_values = filtered_parent.loc[good_parent_value_index][parent_primary_key]
        invalid_rows = (
            data[child_table_name][child_foreign_key]
            .merge(
                good_parent_values,
                left_on=child_foreign_key,
                right_on=parent_primary_key,
                how='left',
                indicator=indicator_col,
            )
            .set_index(data[child_table_name].index)[indicator_col]
            == 'left_only'
        )
        table_to_valid_rows[child_table_name][invalid_rows] = False

    return table_to_valid_rows


def _validate_foreign_to_foreign_key_input(columns, foreign_key_generation):
    """Validates a list of foreign key specifications.

    Args:
        columns (list[dict]):
            A list of dictionaries, each specifying a foreign key that are all connected.
            Each dictionary should have the keys:
                - 'table_name' (str): The name of the table.
                - 'foreign_key' (str or tuple[str]): The foreign key column(s).
        foreign_key_generation (str):
            Method to use to generate new foreign key values. Must be one of ['new', 'reuse'].

    Raises:
        ValueError:
            If the ``columns`` value is not instance of list or dictionaries do not
            contain the right inputs, or if ``foreign_key_generation`` value is not a string or
            is an invalid option.
    """
    expected_length = None

    if not isinstance(columns, list):
        raise ValueError('columns must be a list of dictionaries')

    for entry in columns:
        if not isinstance(entry, dict):
            raise ValueError('Each entry in columns must be a dictionary')

        table_name = entry.get('table_name')
        foreign_key = entry.get('foreign_key')

        if 'table_name' not in entry or not isinstance(table_name, str):
            raise ValueError("Each dictionary must have a 'table_name' key with a string value")

        if 'foreign_key' not in entry:
            raise ValueError("Each dictionary must have a 'foreign_key' key")

        if isinstance(foreign_key, str):
            key_columns = [foreign_key]
        elif isinstance(foreign_key, tuple) and all(isinstance(col, str) for col in foreign_key):
            key_columns = list(foreign_key)
        else:
            raise ValueError("'foreign_key' must be a string or a tuple of strings")

        if expected_length is None:
            expected_length = len(key_columns)

        elif len(key_columns) != expected_length:
            raise ValueError(
                'All foreign key entries must have the same number of columns. '
                f"Entry for table '{table_name}' has {len(key_columns)} columns, "
                f'expected {expected_length}.'
            )

        if not isinstance(foreign_key_generation, str):
            raise ValueError('`foreign_key_generation` must be a string.')

        if foreign_key_generation not in ['new', 'reuse']:
            raise ValueError(
                f"Unrecognized `foreign_key_generation` value '{foreign_key_generation}'. "
                "Must be one of ['new', 'reuse']."
            )


def _get_primary_key(metadata, table_name):
    """Return the primary key of a table, as it is written in the metadata.

    Args:
        metadata (dict):
            The multi table metadata.
        table_name (str):
            The name of the table to get the primary key of.

    Returns:
        str:
            The name of the primary key column.

    Raises:
        ConstraintNotApplicableError:
            If the metadata does not give a primary key for the table.
    """
    tables_metadata = (metadata or {}).get('tables', {})
    primary_key = tables_metadata.get(table_name, {}).get('primary_key')
    if not isinstance(primary_key, str):
        raise ConstraintNotApplicableError(
            f"The table '{table_name}' does not have a primary key in the metadata."
        )

    return primary_key


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


def _format_invalid_values_string(invalid_values, num_values):
    """Convert ``invalid_values`` into a string of invalid values.

    Args:
        invalid_values (pd.DataFrame, set):
            Object of values to be converted into string.
        num_values (int):
            Maximum number of values of the object to show.

    Returns:
        str:
            A stringified version of the object.
    """
    if isinstance(invalid_values, pd.DataFrame):
        if len(invalid_values) > num_values:
            return f'{invalid_values.head(num_values)}\n+{len(invalid_values) - num_values} more'

    if isinstance(invalid_values, set):
        invalid_values = sorted(invalid_values, key=lambda x: str(x))
        if len(invalid_values) > num_values:
            extra_missing_values = [f'+ {len(invalid_values) - num_values} more']
            return f'{invalid_values[:num_values] + extra_missing_values}'

    return f'{invalid_values}'


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
