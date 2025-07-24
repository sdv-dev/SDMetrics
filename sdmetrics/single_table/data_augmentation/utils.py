"""Utils method for data augmentation metrics."""

from sdmetrics._utils_metadata import _validate_single_table_metadata
from sdmetrics.single_table.utils import (
    _validate_classifier,
    _validate_data_and_metadata,
    _validate_prediction_column_name,
    _validate_tables,
)


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

    synthetic_labels = set(synthetic_data[prediction_column_name].unique())
    real_labels = set(real_training_data[prediction_column_name].unique())
    if not synthetic_labels.issubset(real_labels):
        to_print = "', '".join(sorted(synthetic_labels - real_labels))
        raise ValueError(
            f'The `{prediction_column_name}` column must have the same values in the real '
            'and synthetic data. The following values are present in the synthetic data and'
            f" not the real data: '{to_print}'"
        )
