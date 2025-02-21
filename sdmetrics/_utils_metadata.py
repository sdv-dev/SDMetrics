import warnings

import pandas as pd

MODELABLE_SDTYPES = ('numerical', 'datetime', 'categorical', 'boolean')


def _validate_metadata_dict(metadata):
    """Validate the metadata type."""
    if not isinstance(metadata, dict):
        raise TypeError(
            f"Expected a dictionary but received a '{type(metadata).__name__}' instead."
            " For SDV metadata objects, please use the 'to_dict' function to convert it"
            ' to a dictionary.'
        )


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


@handle_single_and_multi_table
def _convert_datetime_columns(data, metadata):
    """Convert datetime columns to datetime type."""
    columns_missing_datetime_format = []
    for column in metadata['columns']:
        if metadata['columns'][column]['sdtype'] == 'datetime':
            is_datetime = pd.api.types.is_datetime64_any_dtype(data[column])
            if not is_datetime:
                datetime_format = metadata['columns'][column].get('format')
                if datetime_format:
                    data[column] = pd.to_datetime(data[column], format=datetime_format)
                else:
                    columns_missing_datetime_format.append(column)
                    data[column] = pd.to_datetime(data[column])

    if columns_missing_datetime_format:
        columns_to_print = "', '".join(columns_missing_datetime_format)
        warnings.warn(
            f'No `datetime_format` provided in the metadata when trying to convert the columns'
            f" '{columns_to_print}' to datetime. The format will be inferred, but it may not"
            ' be accurate.',
            UserWarning,
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
            f"Some columns ('{columns_to_print}') are not present in the metadata."
            'They will not be included for further evaluation.',
            UserWarning,
        )
    elif extra_metadata_columns:
        columns_to_print = "', '".join(sorted(extra_metadata_columns))
        warnings.warn(
            f"Some columns ('{columns_to_print}') are in the metadata but they are not "
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


def _process_data_with_metadata_ml_efficacy_metrics(
    real_training_data, synthetic_data, real_validation_data, metadata
):
    """Process the data for ML efficacy metrics according to the metadata."""
    real_training_data = _process_data_with_metadata(real_training_data, metadata, True)
    synthetic_data = _process_data_with_metadata(synthetic_data, metadata, True)
    real_validation_data = _process_data_with_metadata(real_validation_data, metadata, True)

    return real_training_data, synthetic_data, real_validation_data
