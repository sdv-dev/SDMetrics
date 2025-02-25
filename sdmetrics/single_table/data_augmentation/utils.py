"""Utils method for data augmentation metrics."""

import pandas as pd

from sdmetrics._utils_metadata import _process_data_with_metadata, _validate_single_table_metadata


def _validate_tables(real_training_data, synthetic_data, real_validation_data):
    """Validate the tables of the Data Augmentation metrics."""
    tables = [real_training_data, synthetic_data, real_validation_data]
    if any(not isinstance(table, pd.DataFrame) for table in tables):
        raise ValueError(
            '`real_training_data`, `synthetic_data` and `real_validation_data` must be '
            'pandas DataFrames.'
        )


def _validate_prediction_column_name(prediction_column_name):
    """Validate the prediction column name of the Data Augmentation metrics."""
    if not isinstance(prediction_column_name, str):
        raise TypeError('`prediction_column_name` must be a string.')


def _validate_classifier(classifier):
    """Validate the classifier of the Data Augmentation metrics."""
    if classifier is not None and not isinstance(classifier, str):
        raise TypeError('`classifier` must be a string or None.')

    if classifier != 'XGBoost':
        raise ValueError('Currently only `XGBoost` is supported as classifier.')


def _validate_fixed_recall_value(fixed_recall_value):
    """Validate the fixed recall value of the Data Augmentation metrics."""
    if not isinstance(fixed_recall_value, (int, float)) or not (0 < fixed_recall_value < 1):
        raise TypeError('`fixed_recall_value` must be a float in the range (0, 1).')


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
    _validate_tables(real_training_data, synthetic_data, real_validation_data)
    _validate_single_table_metadata(metadata)
    _validate_prediction_column_name(prediction_column_name)
    _validate_classifier(classifier)
    _validate_fixed_recall_value(fixed_recall_value)


def _validate_data_and_metadata(
    real_training_data,
    synthetic_data,
    real_validation_data,
    metadata,
    prediction_column_name,
    minority_class_label,
):
    """Validate the data and metadata of the Data Augmentation metrics."""
    if prediction_column_name not in metadata['columns']:
        raise ValueError(
            f'The column `{prediction_column_name}` is not described in the metadata.'
            ' Please update your metadata.'
        )

    if metadata['columns'][prediction_column_name]['sdtype'] not in ('categorical', 'boolean'):
        raise ValueError(
            f'The column `{prediction_column_name}` must be either categorical or boolean.'
            ' Please update your metadata.'
        )

    if minority_class_label not in real_training_data[prediction_column_name].unique():
        raise ValueError(
            f'The value `{minority_class_label}` is not present in the column '
            f'`{prediction_column_name}` for the real training data.'
        )

    if minority_class_label not in real_validation_data[prediction_column_name].unique():
        raise ValueError(
            f"The metric can't be computed because the value `{minority_class_label}` "
            f'is not present in the column `{prediction_column_name}` for the real validation data.'
            ' The `precision` and `recall` are undefined for this case.'
        )

    synthetic_labels = set(synthetic_data[prediction_column_name].unique())
    real_labels = set(real_training_data[prediction_column_name].unique())
    if not synthetic_labels.issubset(real_labels):
        to_print = "', '".join(sorted(synthetic_labels - real_labels))
        raise ValueError(
            f'The ``{prediction_column_name}`` column must have the same values in the real '
            'and synthetic data. The following values are present in the synthetic data and'
            f" not the real data: '{to_print}'"
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


def _process_data_with_metadata_ml_efficacy_metrics(
    real_training_data, synthetic_data, real_validation_data, metadata
):
    """Process the data for ML efficacy metrics according to the metadata."""
    real_training_data = _process_data_with_metadata(real_training_data, metadata, True)
    synthetic_data = _process_data_with_metadata(synthetic_data, metadata, True)
    real_validation_data = _process_data_with_metadata(real_validation_data, metadata, True)

    return real_training_data, synthetic_data, real_validation_data
