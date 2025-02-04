import re
from copy import deepcopy
from unittest.mock import patch

import pandas as pd
import pytest

from sdmetrics.single_table.data_augmentation.utils import (
    _validate_data_and_metadata,
    _validate_inputs,
    _validate_parameters,
)


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
        'minority_class_label': 1,
    }
    expected_message_missing_prediction_column = re.escape(
        'The column `target` is not described in the metadata. Please update your metadata.'
    )
    expected_message_sdtype = re.escape(
        'The column `target` must be either categorical or boolean. Please update your metadata.'
    )
    expected_message_column_missmatch = re.escape(
        '`real_training_data`, `synthetic_data` and `real_validation_data` must have the '
        'same columns and must match the columns described in the metadata.'
    )
    expected_message_value = re.escape(
        'The value `1` is not present in the column `target` for the real training data.'
    )
    expected_warning = re.escape(
        'The value `1` is not present in the column `target` for the real validation data.'
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

    wrong_column_metadata = deepcopy(inputs)
    wrong_column_metadata['metadata']['columns'].update({'new_column': {'sdtype': 'categorical'}})
    with pytest.raises(ValueError, match=expected_message_column_missmatch):
        _validate_data_and_metadata(**wrong_column_metadata)

    wrong_column_data = deepcopy(inputs)
    wrong_column_data['real_training_data'] = pd.DataFrame({'new_column': [1, 0, 0]})
    with pytest.raises(ValueError, match=expected_message_column_missmatch):
        _validate_data_and_metadata(**wrong_column_data)

    missing_minority_class_label = deepcopy(inputs)
    missing_minority_class_label['real_training_data'] = pd.DataFrame({'target': [0, 0, 0]})
    with pytest.raises(ValueError, match=expected_message_value):
        _validate_data_and_metadata(**missing_minority_class_label)

    missing_minority_class_label_validation = deepcopy(inputs)
    missing_minority_class_label_validation['real_validation_data'] = pd.DataFrame({
        'target': [0, 0, 0]
    })
    with pytest.warns(UserWarning, match=expected_warning):
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
