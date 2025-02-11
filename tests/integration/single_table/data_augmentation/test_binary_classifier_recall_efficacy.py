import re

import numpy as np
import pytest

from sdmetrics.demos import load_demo
from sdmetrics.single_table.data_augmentation import BinaryClassifierRecallEfficacy


class TestBinaryClassifierRecallEfficacy:
    def test_end_to_end(self):
        """Test the metric end-to-end."""
        # Setup
        np.random.seed(0)
        real_data, synthetic_data, metadata = load_demo(modality='single_table')
        mask_validation = np.random.rand(len(real_data)) < 0.8
        real_training = real_data[mask_validation]
        real_validation = real_data[~mask_validation]
        expected_keys_classifier = {
            'precision_score_training',
            'precision_score_validation',
            'recall_score_validation',
            'prediction_counts_validation',
        }
        expected_keys_confusion_matrix = {
            'true_positive',
            'false_positive',
            'true_negative',
            'false_negative',
        }
        expected_keys_params = {
            'prediction_column_name',
            'minority_class_label',
            'classifier',
            'fixed_precision_value',
        }

        # Run
        score_breakdown = BinaryClassifierRecallEfficacy.compute_breakdown(
            real_training_data=real_training,
            synthetic_data=synthetic_data,
            real_validation_data=real_validation,
            metadata=metadata,
            prediction_column_name='gender',
            minority_class_label='F',
            classifier='XGBoost',
            fixed_precision_value=0.8,
        )

        score = BinaryClassifierRecallEfficacy.compute(
            real_training_data=real_training,
            synthetic_data=synthetic_data,
            real_validation_data=real_validation,
            metadata=metadata,
            prediction_column_name='gender',
            minority_class_label='F',
            classifier='XGBoost',
            fixed_precision_value=0.8,
        )

        # Assert
        assert score_breakdown['real_data_baseline'].keys() == expected_keys_classifier
        assert (
            score_breakdown['real_data_baseline']['prediction_counts_validation'].keys()
            == expected_keys_confusion_matrix
        )
        assert (
            score_breakdown['augmented_data']['prediction_counts_validation'].keys()
            == expected_keys_confusion_matrix
        )
        assert score_breakdown['augmented_data'].keys() == expected_keys_classifier
        assert score_breakdown['parameters'].keys() == expected_keys_params
        assert score == score_breakdown['score']

    def test_with_no_minority_class_in_validation(self):
        """Test the metric when the minority class is not present in the validation data."""
        # Setup
        np.random.seed(0)
        real_data, synthetic_data, metadata = load_demo(modality='single_table')
        mask_validation = np.random.rand(len(real_data)) < 0.8
        real_training = real_data[mask_validation]
        real_validation = real_data[~mask_validation]
        real_validation['gender'] = 'M'
        expected_error = re.escape(
            "The metric can't be computed because the value `F` is not present in the column "
            '`gender` for the real validation data. The `precision` and `recall` are undefined'
            ' for this case.'
        )

        # Run and Assert
        with pytest.raises(ValueError, match=expected_error):
            BinaryClassifierRecallEfficacy.compute(
                real_training_data=real_training,
                synthetic_data=synthetic_data,
                real_validation_data=real_validation,
                metadata=metadata,
                prediction_column_name='gender',
                minority_class_label='F',
                classifier='XGBoost',
                fixed_precision_value=0.8,
            )

    def test_with_nan_target_column(self):
        """Test the metric when the target column has NaN values."""
        # Setup
        np.random.seed(35)
        real_data, synthetic_data, metadata = load_demo(modality='single_table')
        mask_validation = np.random.rand(len(real_data)) < 0.8
        real_training = real_data[mask_validation].reset_index(drop=True)
        real_validation = real_data[~mask_validation].reset_index(drop=True)
        real_training.loc[:3, 'gender'] = np.nan
        real_validation.loc[:5, 'gender'] = np.nan

        # Run
        result_breakdown = BinaryClassifierRecallEfficacy.compute_breakdown(
            real_training_data=real_training,
            synthetic_data=synthetic_data,
            real_validation_data=real_validation,
            metadata=metadata,
            prediction_column_name='gender',
            minority_class_label='F',
            classifier='XGBoost',
            fixed_precision_value=0.8,
        )

        # Assert
        assert result_breakdown['score'] in (0, 0.07692307692307698)

    def test_with_minority_being_majority(self):
        """Test the metric when the minority class is the majority class."""
        # Setup
        np.random.seed(0)
        real_data, synthetic_data, metadata = load_demo(modality='single_table')
        mask_validation = np.random.rand(len(real_data)) < 0.8
        real_training = real_data[mask_validation]
        real_validation = real_data[~mask_validation]

        # Run
        score = BinaryClassifierRecallEfficacy.compute(
            real_training_data=real_training,
            synthetic_data=synthetic_data,
            real_validation_data=real_validation,
            metadata=metadata,
            prediction_column_name='gender',
            minority_class_label='F',
            classifier='XGBoost',
            fixed_precision_value=0.8,
        )

        # Assert
        assert score == 0

    def test_with_multi_class(self):
        """Test the metric with multi-class classification.

        The `high_spec` column has 3 values `Commerce`, `Science`, and `Arts`.
        """
        # Setup
        np.random.seed(0)
        real_data, synthetic_data, metadata = load_demo(modality='single_table')
        mask_validation = np.random.rand(len(real_data)) < 0.8
        real_training = real_data[mask_validation]
        real_validation = real_data[~mask_validation]

        # Run
        score_breakdown = BinaryClassifierRecallEfficacy.compute_breakdown(
            real_training_data=real_training,
            synthetic_data=synthetic_data,
            real_validation_data=real_validation,
            metadata=metadata,
            prediction_column_name='high_spec',
            minority_class_label='Science',
            classifier='XGBoost',
            fixed_precision_value=0.8,
        )

        # Assert
        assert score_breakdown['score'] in (0, 0.07692307692307687)
