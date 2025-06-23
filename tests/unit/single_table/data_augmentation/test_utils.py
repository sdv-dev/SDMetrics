import re
from copy import deepcopy
from unittest.mock import call, patch

import pandas as pd
import pytest

from sdmetrics.single_table.data_augmentation.utils import (
    _validate_data_and_metadata,
    _validate_inputs,
    _validate_parameters,
)
from sdmetrics.single_table.utils import _process_data_with_metadata_ml_efficacy_metrics


def test__validate_parameters():
    """Test the ``_validate_parameters`` method."""
    # Setup
    expected_message_dataframes = re.escape(
        '`real_training_data`, `synthetic_data` and `real_validation_data` must be'
        ' pandas DataFrames.'
    )
    expected_message_metadata = re.escape(
        "Expected a dictionary but received a 'list' instead."
        " For SDV metadata objects, please use the 'to_dict' function to convert it"
        ' to a dictionary.'
    )
    expected_message_prediction_column_name = re.escape(
        '`prediction_column_name` must be a string.'
    )
    expected_message_classifier = re.escape('`classifier` must be a string or None.')
    expected_message_classifier_value = re.escape(
        'Currently only `XGBoost` is supported as classifier.'
    )
    expected_message_fixed_recall_value = re.escape(
        '`fixed_recall_value` must be a float in the range (0, 1).'
    )
    inputs = {
        'real_training_data': pd.DataFrame({'target': [1, 0, 0]}),
        'synthetic_data': pd.DataFrame({'target': [1, 0, 0]}),
        'real_validation_data': pd.DataFrame({'target': [1, 0, 0]}),
        'metadata': {'columns': {'target': {'sdtype': 'categorical'}}},
        'prediction_column_name': 'target',
        'classifier': 'XGBoost',
        'fixed_recall_value': 0.9,
    }

    # Run and Assert
    _validate_parameters(**inputs)
    wrong_inputs_dataframes = deepcopy(inputs)
    wrong_inputs_dataframes['real_training_data'] = 'wrong'
    with pytest.raises(ValueError, match=expected_message_dataframes):
        _validate_parameters(**wrong_inputs_dataframes)

    wrong_inputs_metadata = deepcopy(inputs)
    wrong_inputs_metadata['metadata'] = []
    with pytest.raises(TypeError, match=expected_message_metadata):
        _validate_parameters(**wrong_inputs_metadata)

    wrong_inputs_prediction_column_name = deepcopy(inputs)
    wrong_inputs_prediction_column_name['prediction_column_name'] = 1
    with pytest.raises(TypeError, match=expected_message_prediction_column_name):
        _validate_parameters(**wrong_inputs_prediction_column_name)

    wrong_inputs_classifier_type = deepcopy(inputs)
    wrong_inputs_classifier_type['classifier'] = 1
    with pytest.raises(TypeError, match=expected_message_classifier):
        _validate_parameters(**wrong_inputs_classifier_type)

    wrong_inputs_classifier = deepcopy(inputs)
    wrong_inputs_classifier['classifier'] = 'LogisticRegression'
    with pytest.raises(ValueError, match=expected_message_classifier_value):
        _validate_parameters(**wrong_inputs_classifier)

    wrong_inputs_fixed_recall_value_type = deepcopy(inputs)
    wrong_inputs_fixed_recall_value_type['fixed_recall_value'] = '0.9'
    with pytest.raises(TypeError, match=expected_message_fixed_recall_value):
        _validate_parameters(**wrong_inputs_fixed_recall_value_type)

    wrong_inputs_fixed_recall_value = deepcopy(inputs)
    wrong_inputs_fixed_recall_value['fixed_recall_value'] = 1.2
    with pytest.raises(TypeError, match=expected_message_fixed_recall_value):
        _validate_parameters(**wrong_inputs_fixed_recall_value)


def test__validate_data_and_metadata():
    """Test the ``_validate_data_and_metadata`` method."""
    # Setup
    inputs = {
        'real_training_data': pd.DataFrame({'target': [1, 0, 0]}),
        'synthetic_data': pd.DataFrame({'target': [1, 0, 0]}),
        'real_validation_data': pd.DataFrame({'target': [1, 0, 0]}),
        'metadata': {'columns': {'target': {'sdtype': 'categorical'}}},
        'prediction_column_name': 'target',
        'prediction_column_label': 1,
    }
    expected_message_missing_prediction_column = re.escape(
        'The column `target` is not described in the metadata. Please update your metadata.'
    )
    expected_message_sdtype = re.escape(
        'The column `target` must be either categorical or boolean. Please update your metadata.'
    )
    expected_message_value = re.escape(
        'The value `1` is not present in the column `target` for the real training data.'
    )
    expected_error_missing_minority = re.escape(
        "The metric can't be computed because the value `1` is not present in "
        'the column `target` for the real validation data. The `precision` and `recall`'
        ' are undefined for this case.'
    )

    # Run and Assert
    _validate_data_and_metadata(**inputs)
    missing_prediction_column = deepcopy(inputs)
    missing_prediction_column['metadata']['columns'].pop('target')
    with pytest.raises(ValueError, match=expected_message_missing_prediction_column):
        _validate_data_and_metadata(**missing_prediction_column)

    wrong_inputs_sdtype = deepcopy(inputs)
    wrong_inputs_sdtype['metadata']['columns']['target']['sdtype'] = 'numerical'
    with pytest.raises(ValueError, match=expected_message_sdtype):
        _validate_data_and_metadata(**wrong_inputs_sdtype)

    missing_minority_class_label = deepcopy(inputs)
    missing_minority_class_label['real_training_data'] = pd.DataFrame({'target': [0, 0, 0]})
    with pytest.raises(ValueError, match=expected_message_value):
        _validate_data_and_metadata(**missing_minority_class_label)

    missing_minority_class_label_validation = deepcopy(inputs)
    missing_minority_class_label_validation['real_validation_data'] = pd.DataFrame({
        'target': [0, 0, 0]
    })
    with pytest.raises(ValueError, match=expected_error_missing_minority):
        _validate_data_and_metadata(**missing_minority_class_label_validation)


@patch('sdmetrics.single_table.data_augmentation.utils._validate_parameters')
@patch('sdmetrics.single_table.data_augmentation.utils._validate_data_and_metadata')
def test__validate_inputs_mock(mock_validate_data_and_metadata, mock_validate_parameters):
    """Test the ``validate_inputs`` method."""
    # Setup
    real_training_data = pd.DataFrame({'target': [1, 0, 0]})
    synthetic_data = pd.DataFrame({'target': [1, 0, 0]})
    real_validation_data = pd.DataFrame({'target': [1, 0, 0]})
    metadata = {'columns': {'target': {'sdtype': 'categorical'}}}
    prediction_column_name = 'target'
    minority_class_label = 1
    classifier = 'XGBoost'
    fixed_recall_value = 0.9

    # Run
    _validate_inputs(
        real_training_data,
        synthetic_data,
        real_validation_data,
        metadata,
        prediction_column_name,
        minority_class_label,
        classifier,
        fixed_recall_value,
    )

    # Assert
    mock_validate_data_and_metadata.assert_called_once_with(
        real_training_data,
        synthetic_data,
        real_validation_data,
        metadata,
        prediction_column_name,
        minority_class_label,
    )
    mock_validate_parameters.assert_called_once_with(
        real_training_data,
        synthetic_data,
        real_validation_data,
        metadata,
        prediction_column_name,
        classifier,
        fixed_recall_value,
    )

    expected_error_synthetic_wrong_label = re.escape(
        'The `target` column must have the same values in the real and synthetic data. '
        'The following values are present in the synthetic data and not the real'
        " data: 'wrong_1', 'wrong_2'"
    )
    wrong_synthetic_label = pd.DataFrame({'target': [0, 1, 'wrong_1', 'wrong_2']})
    with pytest.raises(ValueError, match=expected_error_synthetic_wrong_label):
        _validate_inputs(
            real_training_data,
            wrong_synthetic_label,
            real_validation_data,
            metadata,
            prediction_column_name,
            minority_class_label,
            classifier,
            fixed_recall_value,
        )


@patch('sdmetrics.single_table.utils._process_data_with_metadata')
def test__process_data_with_metadata_ml_efficacy_metrics(mock_process_data_with_metadata):
    """Test the ``_process_data_with_metadata_ml_efficacy_metrics`` method."""
    # Setup
    mock_process_data_with_metadata.side_effect = lambda data, metadata, x: data
    real_training_data = pd.DataFrame({
        'numerical': [1, 2, 3],
        'categorical': ['a', 'b', 'c'],
    })
    synthetic_data = pd.DataFrame({
        'numerical': [4, 5, 6],
        'categorical': ['a', 'b', 'c'],
    })
    real_validation_data = pd.DataFrame({
        'numerical': [7, 8, 9],
        'categorical': ['a', 'b', 'c'],
    })
    metadata = {
        'columns': {
            'numerical': {'sdtype': 'numerical'},
            'categorical': {'sdtype': 'categorical'},
        }
    }

    # Run
    result = _process_data_with_metadata_ml_efficacy_metrics(
        real_training_data, synthetic_data, real_validation_data, metadata
    )

    # Assert
    pd.testing.assert_frame_equal(result[0], real_training_data)
    pd.testing.assert_frame_equal(result[1], synthetic_data)
    pd.testing.assert_frame_equal(result[2], real_validation_data)
    mock_process_data_with_metadata.assert_has_calls([
        call(real_training_data, metadata, True),
        call(synthetic_data, metadata, True),
        call(real_validation_data, metadata, True),
    ])
