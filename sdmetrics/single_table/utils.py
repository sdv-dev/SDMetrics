"""Shared utility methods for single table metrics."""

import pandas as pd

from sdmetrics._utils_metadata import _process_data_with_metadata


def _validate_tables(real_training_data, synthetic_data, real_validation_data):
    """Validate the tables of the single table metrics."""
    tables = [real_training_data, synthetic_data, real_validation_data]
    if any(not isinstance(table, pd.DataFrame) for table in tables):
        raise ValueError(
            '`real_training_data`, `synthetic_data` and `real_validation_data` must be '
            'pandas DataFrames.'
        )


def _validate_prediction_column_name(prediction_column_name):
    """Validate the prediction column name of the single table metrics."""
    if not isinstance(prediction_column_name, str):
        raise TypeError('`prediction_column_name` must be a string.')


def _validate_sensitive_column_name(sensitive_column_name):
    """Validate the sensitive column name of the single table metrics."""
    if not isinstance(sensitive_column_name, str):
        raise TypeError('`sensitive_column_name` must be a string.')


def _validate_classifier(classifier):
    """Validate the classifier of the single table metrics."""
    if classifier is not None and not isinstance(classifier, str):
        raise TypeError('`classifier` must be a string or None.')

    if classifier != 'XGBoost':
        raise ValueError('Currently only `XGBoost` is supported as classifier.')


def _validate_required_columns(dataframes_dict, required_columns):
    """Validate that required columns exist in all datasets.

    Args:
        dataframes_dict (dict): Dictionary mapping dataset names to DataFrames
        required_columns (list): List of required column names

    Raises:
        ValueError: If any required columns are missing from any dataset
    """
    for df_name, df in dataframes_dict.items():
        missing_cols = [col for col in required_columns if col not in df.columns]
        if missing_cols:
            raise ValueError(f'Missing columns in {df_name}: {missing_cols}')


def _validate_column_values_exist(dataframes_dict, column_value_pairs):
    """Validate that specified values exist in specified columns across all datasets.

    Args:
        dataframes_dict (dict): Dictionary mapping dataset names to DataFrames
        column_value_pairs (list): List of (column_name, value) tuples to validate

    Raises:
        ValueError: If any specified values don't exist in the specified columns
    """
    for df_name, df in dataframes_dict.items():
        for column_name, value in column_value_pairs:
            column_values = df[column_name]
            value_exists = (pd.isna(value) and column_values.isna().any()) or (
                value in column_values.to_numpy()
            )
            if not value_exists:
                raise ValueError(f"Value '{value}' not found in {df_name}['{column_name}']")


def _validate_column_consistency(real_training_data, synthetic_data, real_validation_data):
    """Validate that validation data has same columns as training data.

    Args:
        real_training_data (pandas.DataFrame): Real training data
        synthetic_data (pandas.DataFrame): Synthetic data
        real_validation_data (pandas.DataFrame): Real validation data

    Raises:
        ValueError: If column sets don't match
    """
    if set(real_validation_data.columns) != set(synthetic_data.columns) or set(
        real_validation_data.columns
    ) != set(real_training_data.columns):
        raise ValueError(
            'real_validation_data must have the same columns as synthetic_data and '
            'real_training_data'
        )


def _validate_data_and_metadata(
    real_training_data,
    synthetic_data,
    real_validation_data,
    metadata,
    prediction_column_name,
    prediction_column_label,
):
    """Validate the data and metadata consistency for single table metrics.

    Args:
        real_training_data (pandas.DataFrame):
            Real training data
        synthetic_data (pandas.DataFrame):
            Synthetic data
        real_validation_data (pandas.DataFrame):
            Real validation data
        metadata (dict):
            Metadata describing the table
        prediction_column_name (str):
            Name of the prediction column
        prediction_column_label:
            The prediction column label to validate

    Raises:
        ValueError: If validation fails
    """
    if prediction_column_name not in metadata.get('columns', {}):
        raise ValueError(
            f'The column `{prediction_column_name}` is not described in the metadata.'
            ' Please update your metadata.'
        )

    column_sdtype = metadata['columns'][prediction_column_name].get('sdtype')
    if column_sdtype not in ('categorical', 'boolean'):
        raise ValueError(
            f'The column `{prediction_column_name}` must be either categorical or boolean.'
            ' Please update your metadata.'
        )

    if prediction_column_label not in real_training_data[prediction_column_name].unique():
        raise ValueError(
            f'The value `{prediction_column_label}` is not present in the column '
            f'`{prediction_column_name}` for the real training data.'
        )

    if prediction_column_label not in real_validation_data[prediction_column_name].unique():
        raise ValueError(
            f"The metric can't be computed because the value `{prediction_column_label}` "
            f'is not present in the column `{prediction_column_name}` for the real validation data.'
            ' The `precision` and `recall` are undefined for this case.'
        )


def _process_data_with_metadata_ml_efficacy_metrics(
    real_training_data, synthetic_data, real_validation_data, metadata
):
    """Process the data for ML efficacy metrics according to the metadata."""
    real_training_data = _process_data_with_metadata(real_training_data, metadata, True)
    synthetic_data = _process_data_with_metadata(synthetic_data, metadata, True)
    real_validation_data = _process_data_with_metadata(real_validation_data, metadata, True)

    return real_training_data, synthetic_data, real_validation_data
