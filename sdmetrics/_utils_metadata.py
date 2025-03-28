import warnings

import pandas as pd

from sdmetrics.utils import is_datetime

MODELABLE_SDTYPES = ('numerical', 'datetime', 'categorical', 'boolean')


def _validate_metadata_dict(metadata):
    """Validate the metadata type."""
    if not isinstance(metadata, dict):
        raise TypeError(
            f"Expected a dictionary but received a '{type(metadata).__name__}' instead."
            " For SDV metadata objects, please use the 'to_dict' function to convert it"
            ' to a dictionary.'
        )


def _validate_single_table_metadata(metadata):
    """Validate the metadata for a single table."""
    _validate_metadata_dict(metadata)
    if 'columns' not in metadata:
        raise ValueError(
            "Single-table metadata must include a 'columns' key that maps column names"
            ' to their corresponding information.'
        )


def _validate_multi_table_metadata(metadata):
    """Validate the metadata for multiple tables."""
    _validate_metadata_dict(metadata)
    if 'tables' not in metadata:
        raise ValueError(
            "Multi-table metadata must include a 'tables' key that maps table names"
            ' to their respective metadata.'
        )
    for table_name, table_metadata in metadata['tables'].items():
        try:
            _validate_single_table_metadata(table_metadata)
        except ValueError as e:
            raise ValueError(f"Error in table '{table_name}': {str(e)}")


def _validate_metadata(metadata):
    """Validate the metadata."""
    _validate_metadata_dict(metadata)
    if ('columns' not in metadata) and ('tables' not in metadata):
        raise ValueError(
            "Metadata must include either a 'columns' key for single-table metadata"
            " or a 'tables' key for multi-table metadata."
        )

    if 'tables' in metadata:
        _validate_multi_table_metadata(metadata)


def handle_single_and_multi_table(single_table_func):
    """Decorator to handle both single and multi table functions."""

    def wrapper(data, metadata):
        if isinstance(data, pd.DataFrame):
            return single_table_func(data, metadata)

        result = {}
        for table_name in data:
            result[table_name] = single_table_func(data[table_name], metadata['tables'][table_name])

        return result

    return wrapper


def _convert_datetime_column(column_name, column_data, column_metadata):
    if is_datetime(column_data):
        return column_data

    datetime_format = column_metadata.get('datetime_format')
    if datetime_format is None:
        raise ValueError(
            f"Datetime column '{column_name}' does not have a specified 'datetime_format'. "
            'Please add a the required datetime_format to the metadata or convert this column '
            "to 'pd.datetime' to bypass this requirement."
        )

    try:
        pd.to_datetime(column_data, format=datetime_format)
    except Exception as e:
        raise ValueError(f"Error converting column '{column_name}' to timestamp: {e}")

    return pd.to_datetime(column_data, format=datetime_format)


@handle_single_and_multi_table
def _convert_datetime_columns(data, metadata):
    """Convert datetime columns to datetime type."""
    for column in metadata['columns']:
        if metadata['columns'][column]['sdtype'] == 'datetime':
            data[column] = _convert_datetime_column(
                column, data[column], metadata['columns'][column]
            )

    return data


@handle_single_and_multi_table
def _remove_missing_columns_metadata(data, metadata):
    """Remove columns that are not present in the metadata."""
    columns_in_metadata = set(metadata['columns'].keys())
    columns_in_data = set(data.columns)
    columns_to_remove = columns_in_data - columns_in_metadata
    extra_metadata_columns = columns_in_metadata - columns_in_data
    if columns_to_remove:
        columns_to_print = "', '".join(sorted(columns_to_remove))
        warnings.warn(
            f"The columns ('{columns_to_print}') are not present in the metadata. "
            'They will not be included for further evaluation.',
            UserWarning,
        )
    elif extra_metadata_columns:
        columns_to_print = "', '".join(sorted(extra_metadata_columns))
        warnings.warn(
            f"The columns ('{columns_to_print}') are in the metadata but they are not "
            'present in the data.',
            UserWarning,
        )

    data = data.drop(columns=columns_to_remove)
    column_intersection = [column for column in data.columns if column in metadata['columns']]

    return data[column_intersection]


@handle_single_and_multi_table
def _remove_non_modelable_columns(data, metadata):
    """Remove columns that are not modelable.

    All modelable columns are numerical, datetime, categorical, or boolean sdtypes.
    """
    columns_modelable = []
    for column in metadata['columns']:
        column_sdtype = metadata['columns'][column]['sdtype']
        if column_sdtype in MODELABLE_SDTYPES and column in data.columns:
            columns_modelable.append(column)

    return data[columns_modelable]


def _process_data_with_metadata(data, metadata, keep_modelable_columns_only=False):
    """Process the data according to the metadata."""
    _validate_metadata_dict(metadata)
    data = _convert_datetime_columns(data, metadata)
    data = _remove_missing_columns_metadata(data, metadata)
    if keep_modelable_columns_only:
        data = _remove_non_modelable_columns(data, metadata)

    return data
