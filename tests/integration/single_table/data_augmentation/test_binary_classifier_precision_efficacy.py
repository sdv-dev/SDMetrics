import re
from unittest.mock import patch

import numpy as np
import pytest
from xgboost import XGBClassifier

from sdmetrics.demos import load_demo
from sdmetrics.single_table.data_augmentation import BinaryClassifierPrecisionEfficacy


@pytest.mark.usefixtures('xgboost_init')
class TestBinaryClassifierPrecisionEfficacy:
    @pytest.fixture(autouse=True)
    def apply_xgboost_patch(self, xgboost_init):
        with patch.object(XGBClassifier, '__init__', autospec=True, side_effect=xgboost_init):
            yield

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
            fixed_recall_value=0.8,
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
                'recall_score_validation': 0.8461538461538461,
                'precision_score_validation': 0.4583333333333333,
                'prediction_counts_validation': {
                    'true_positive': 11,
                    'false_positive': 13,
                    'true_negative': 12,
                    'false_negative': 2,
                },
            },
            'augmented_data': {
                'recall_score_training': 0.8776978417266187,
                'recall_score_validation': 0.8461538461538461,
                'precision_score_validation': 0.3548387096774194,
                'prediction_counts_validation': {
                    'true_positive': 11,
                    'false_positive': 20,
                    'true_negative': 5,
                    'false_negative': 2,
                },
            },
            'parameters': {
                'prediction_column_name': 'gender',
                'minority_class_label': 'F',
                'classifier': 'XGBoost',
                'fixed_recall_value': 0.8,
            },
            'score': 0.448252688172043,
        }
        assert np.isclose(
            score_breakdown['real_data_baseline']['recall_score_training'], 0.8, atol=0.1
        )
        assert np.isclose(
            score_breakdown['augmented_data']['recall_score_validation'], 0.8, atol=0.1
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
                'recall_score_training': 0.8983050847457628,
                'recall_score_validation': 0.6153846153846154,
                'precision_score_validation': 0.26666666666666666,
                'prediction_counts_validation': {
                    'true_positive': 8,
                    'false_positive': 22,
                    'true_negative': 11,
                    'false_negative': 5,
                },
            },
            'augmented_data': {
                'recall_score_training': 0.8814814814814815,
                'recall_score_validation': 0.6923076923076923,
                'precision_score_validation': 0.2903225806451613,
                'prediction_counts_validation': {
                    'true_positive': 9,
                    'false_positive': 22,
                    'true_negative': 11,
                    'false_negative': 4,
                },
            },
            'parameters': {
                'prediction_column_name': 'gender',
                'minority_class_label': 'F',
                'classifier': 'XGBoost',
                'fixed_recall_value': 0.8,
            },
            'score': 0.5118279569892473,
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
        assert score == 0.448252688172043

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
                'recall_score_training': 0.8205128205128205,
                'recall_score_validation': 0.7692307692307693,
                'precision_score_validation': 0.8333333333333334,
                'prediction_counts_validation': {
                    'true_positive': 10,
                    'false_positive': 2,
                    'true_negative': 23,
                    'false_negative': 3,
                },
            },
            'augmented_data': {
                'recall_score_training': 0.8690476190476191,
                'recall_score_validation': 0.8461538461538461,
                'precision_score_validation': 0.4782608695652174,
                'prediction_counts_validation': {
                    'true_positive': 11,
                    'false_positive': 12,
                    'true_negative': 13,
                    'false_negative': 2,
                },
            },
            'parameters': {
                'prediction_column_name': 'high_spec',
                'minority_class_label': 'Science',
                'classifier': 'XGBoost',
                'fixed_recall_value': 0.8,
            },
            'score': 0.322463768115942,
        }
        assert score_breakdown == expected_score_breakdown
