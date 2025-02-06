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

        # Run
        score_breakdown = BinaryClassifierRecallEfficacy.compute_breakdown(
            real_training_data=real_training,
            synthetic_data=synthetic_data,
            real_validation_data=real_validation,
            metadata=metadata,
            prediction_column_name='gender',
            minority_class_label='F',
            classifier='XGBoost',
            fixed_recall_value=0.8,
        )

        score = BinaryClassifierRecallEfficacy.compute(
            real_training_data=real_training,
            synthetic_data=synthetic_data,
            real_validation_data=real_validation,
            metadata=metadata,
            prediction_column_name='gender',
            minority_class_label='F',
            classifier='XGBoost',
            fixed_recall_value=0.8,
        )

        # Assert
        expected_score_breakdown = {
            'real_data_baseline': {
                'precision_score_training': 0.8076923076923077,
                'recall_score_validation': 0.8461538461538461,
                'precision_score_validation': 0.4230769230769231,
                'prediction_counts_validation': {
                    'true_positive': 11,
                    'false_positive': 15,
                    'true_negative': 10,
                    'false_negative': 2,
                },
            },
            'augmented_data': {
                'precision_score_training': 0.8034682080924855,
                'recall_score_validation': 0.7692307692307693,
                'precision_score_validation': 0.4,
                'prediction_counts_validation': {
                    'true_positive': 10,
                    'false_positive': 15,
                    'true_negative': 10,
                    'false_negative': 3,
                },
            },
            'parameters': {
                'prediction_column_name': 'gender',
                'minority_class_label': 'F',
                'classifier': 'XGBoost',
                'fixed_recall_value': 0.8,
            },
            'score': 0,
        }
        assert np.isclose(
            score_breakdown['real_data_baseline']['precision_score_training'], 0.8, atol=0.1
        )
        assert np.isclose(
            score_breakdown['augmented_data']['precision_score_validation'], 0.44, atol=0.1
        )
        assert score_breakdown == expected_score_breakdown
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
            '`gender` for the real validation data. The `precision`and `recall` are undefined'
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
                fixed_recall_value=0.8,
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
            fixed_recall_value=0.8,
        )

        # Assert
        expected_result = {
            'real_data_baseline': {
                'precision_score_training': 0.8082191780821918,
                'recall_score_validation': 0.6923076923076923,
                'precision_score_validation': 0.391304347826087,
                'prediction_counts_validation': {
                    'true_positive': 9,
                    'false_positive': 14,
                    'true_negative': 19,
                    'false_negative': 4,
                },
            },
            'augmented_data': {
                'precision_score_training': 0.8035714285714286,
                'recall_score_validation': 0.7692307692307693,
                'precision_score_validation': 0.38461538461538464,
                'prediction_counts_validation': {
                    'true_positive': 10,
                    'false_positive': 16,
                    'true_negative': 17,
                    'false_negative': 3,
                },
            },
            'parameters': {
                'prediction_column_name': 'gender',
                'minority_class_label': 'F',
                'classifier': 'XGBoost',
                'fixed_recall_value': 0.8,
            },
            'score': 0.07692307692307698,
        }
        assert result_breakdown == expected_result

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
            fixed_recall_value=0.8,
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
            fixed_recall_value=0.8,
        )

        # Assert
        expected_score_breakdown = {
            'real_data_baseline': {
                'precision_score_training': 0.8041237113402062,
                'recall_score_validation': 0.9230769230769231,
                'precision_score_validation': 0.5,
                'prediction_counts_validation': {
                    'true_positive': 12,
                    'false_positive': 12,
                    'true_negative': 13,
                    'false_negative': 1,
                },
            },
            'augmented_data': {
                'precision_score_training': 0.8,
                'recall_score_validation': 1.0,
                'precision_score_validation': 0.4482758620689655,
                'prediction_counts_validation': {
                    'true_positive': 13,
                    'false_positive': 16,
                    'true_negative': 9,
                    'false_negative': 0,
                },
            },
            'parameters': {
                'prediction_column_name': 'high_spec',
                'minority_class_label': 'Science',
                'classifier': 'XGBoost',
                'fixed_recall_value': 0.8,
            },
            'score': 0.07692307692307687,
        }
        assert score_breakdown == expected_score_breakdown
