"""Utility Functions for Constraints."""

from datetime import datetime

import pandas as pd
import numpy as np

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

