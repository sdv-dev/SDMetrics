"""Utils method for data augmentation metrics."""

import warnings

import pandas as pd


def _validate_parameters(
    real_training_data,
    synthetic_data,
    real_validation_data,
    metadata,
    prediction_column_name,
    classifier,
    fixed_recall_value,
):
    """Validate the parameters of the Data Augmentation metrics."""
    tables = [real_training_data, synthetic_data, real_validation_data]
    if any(not isinstance(table, pd.DataFrame) for table in tables):
        raise ValueError(
            '`real_training_data`, `synthetic_data` and `real_validation_data` must be '
            'pandas DataFrames.'
        )

    if not isinstance(metadata, dict):
        raise TypeError(
            f"Expected a dictionary but received a '{type(metadata).__name__}' instead."
            " For SDV metadata objects, please use the 'to_dict' function to convert it"
            ' to a dictionary.'
        )

    if not isinstance(prediction_column_name, str):
        raise TypeError('`prediction_column_name` must be a string.')

    if classifier is not None and not isinstance(classifier, str):
        raise TypeError('`classifier` must be a string or None.')

    if classifier != 'XGBoost':
        raise ValueError('Currently only `XGBoost` is supported as classifier.')

    if not isinstance(fixed_recall_value, (int, float)) or not (0 < fixed_recall_value < 1):
        raise TypeError('`fixed_recall_value` must be a float in the range (0, 1).')


def _validate_data_and_metadata(
    real_training_data,
    synthetic_data,
    real_validation_data,
    metadata,
    prediction_column_name,
    minority_class_label,
):
    """Validate the data and metadata of the Data Augmentation metrics."""
    if metadata['columns'][prediction_column_name]['stype'] not in ('categorical', 'boolean'):
        raise ValueError(
            f'The column `{prediction_column_name}` must be either categorical or boolean.'
            'Please update your metadata.'
        )

    columns_match = (
        set(real_training_data.columns)
        == set(synthetic_data.columns)
        == set(real_validation_data.columns)
    )
    data_metadata_mismatch = set(metadata['columns'].keys()) != set(real_training_data.columns)
    if not columns_match or data_metadata_mismatch:
        raise ValueError(
            '`real_training_data`, `synthetic_data` and `real_validation_data` must have '
            'the same columns and must match the columns described in the metadata.'
        )

    if minority_class_label not in real_training_data[prediction_column_name].unique():
        raise ValueError(
            f'The value `{minority_class_label}` is not present in the column '
            f'`{prediction_column_name}` for the real training data.'
        )

    if minority_class_label not in real_validation_data[prediction_column_name].unique():
        warnings.warn(
            f'The value `{minority_class_label}` is not present in the column '
            f'`{prediction_column_name}` for the real validation data.'
        )


def _validate_inputs(
    real_training_data,
    synthetic_data,
    real_validation_data,
    metadata,
    prediction_column_name,
    minority_class_label,
    classifier,
    fixed_recall_value,
):
    """Validate the inputs of the Data Augmentation metrics."""
    _validate_parameters(
        real_training_data,
        synthetic_data,
        real_validation_data,
        metadata,
        prediction_column_name,
        classifier,
        fixed_recall_value,
    )
    _validate_data_and_metadata(
        real_training_data,
        synthetic_data,
        real_validation_data,
        metadata,
        prediction_column_name,
        minority_class_label,
    )
