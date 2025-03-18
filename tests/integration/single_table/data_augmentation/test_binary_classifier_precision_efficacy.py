import re

import numpy as np
import pytest

from sdmetrics.demos import load_demo
from sdmetrics.single_table.data_augmentation import BinaryClassifierPrecisionEfficacy


class TestBinaryClassifierPrecisionEfficacy:
    def test_end_to_end(self):
        """Test the metric end-to-end."""
        # Setup
        np.random.seed(0)
        real_data, synthetic_data, metadata = load_demo(modality='single_table')
        mask_validation = np.random.rand(len(real_data)) < 0.8
        real_training = real_data[mask_validation]
        real_validation = real_data[~mask_validation]

        # Run
        score_breakdown = BinaryClassifierPrecisionEfficacy.compute_breakdown(
            real_training_data=real_training,
            synthetic_data=synthetic_data,
            real_validation_data=real_validation,
            metadata=metadata,
            prediction_column_name='gender',
            minority_class_label='F',
        )

        score = BinaryClassifierPrecisionEfficacy.compute(
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
                'recall_score_training': 0.8095238095238095,
                'recall_score_validation': 0.15384615384615385,
                'precision_score_validation': 0.4,
                'prediction_counts_validation': {
                    'true_positive': 2,
                    'false_positive': 3,
                    'true_negative': 22,
                    'false_negative': 11,
                },
            },
            'augmented_data': {
                'recall_score_training': 0.8057553956834532,
                'recall_score_validation': 0.0,
                'precision_score_validation': 0.0,
                'prediction_counts_validation': {
                    'true_positive': 0,
                    'false_positive': 1,
                    'true_negative': 24,
                    'false_negative': 13,
                },
            },
            'parameters': {
                'prediction_column_name': 'gender',
                'minority_class_label': 'F',
                'classifier': 'XGBoost',
                'fixed_recall_value': 0.8,
            },
            'score': 0.3,
        }
        assert np.isclose(
            score_breakdown['real_data_baseline']['recall_score_training'], 0.8, atol=0.1
        )
        assert np.isclose(
            score_breakdown['augmented_data']['recall_score_validation'], 0.1, atol=0.1
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
            '`gender` for the real validation data. The `precision` and `recall` are undefined'
            ' for this case.'
        )

        # Run and Assert
        with pytest.raises(ValueError, match=expected_error):
            BinaryClassifierPrecisionEfficacy.compute(
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
        result_breakdown = BinaryClassifierPrecisionEfficacy.compute_breakdown(
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
                'recall_score_training': 0.8135593220338984,
                'recall_score_validation': 0.23076923076923078,
                'precision_score_validation': 0.42857142857142855,
                'prediction_counts_validation': {
                    'true_positive': 3,
                    'false_positive': 4,
                    'true_negative': 29,
                    'false_negative': 10,
                },
            },
            'augmented_data': {
                'recall_score_training': 0.8,
                'recall_score_validation': 0.23076923076923078,
                'precision_score_validation': 0.6,
                'prediction_counts_validation': {
                    'true_positive': 3,
                    'false_positive': 2,
                    'true_negative': 31,
                    'false_negative': 10,
                },
            },
            'parameters': {
                'prediction_column_name': 'gender',
                'minority_class_label': 'F',
                'classifier': 'XGBoost',
                'fixed_recall_value': 0.8,
            },
            'score': 0.5857142857142857,
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
        score = BinaryClassifierPrecisionEfficacy.compute(
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
        assert score == 0.3

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
        score_breakdown = BinaryClassifierPrecisionEfficacy.compute_breakdown(
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
                'recall_score_training': 0.8076923076923077,
                'recall_score_validation': 0.6923076923076923,
                'precision_score_validation': 0.9,
                'prediction_counts_validation': {
                    'true_positive': 9,
                    'false_positive': 1,
                    'true_negative': 24,
                    'false_negative': 4,
                },
            },
            'augmented_data': {
                'recall_score_training': 0.8035714285714286,
                'recall_score_validation': 0.46153846153846156,
                'precision_score_validation': 1.0,
                'prediction_counts_validation': {
                    'true_positive': 6,
                    'false_positive': 0,
                    'true_negative': 25,
                    'false_negative': 7,
                },
            },
            'parameters': {
                'prediction_column_name': 'high_spec',
                'minority_class_label': 'Science',
                'classifier': 'XGBoost',
                'fixed_recall_value': 0.8,
            },
            'score': 0.55,
        }
        assert score_breakdown == expected_score_breakdown
