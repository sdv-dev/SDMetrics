"""Test for the Binary Classifier Precision Efficacy metrics."""

from unittest.mock import patch

import pandas as pd

from sdmetrics.single_table.data_augmentation.binary_classifier_recall_efficacy import (
    BinaryClassifierRecallEfficacy,
)


class TestBinaryClassifierPrecisionEfficacy:
    def test_class_attributes(self):
        """Test the class attributes."""
        # Setup
        expected_name = 'Binary Classifier Recall Efficacy'
        expected_metric_name = 'recall'

        # Run and Assert
        assert BinaryClassifierRecallEfficacy.name == expected_name
        assert BinaryClassifierRecallEfficacy.metric_name == expected_metric_name

    @patch(
        'sdmetrics.single_table.data_augmentation.base.BaseDataAugmentationMetric.compute_breakdown'
    )
    def test_compute_breakdown(self, mock_compute_breakdown):
        """Test the compute_breakdown method."""
        # Setup
        real_training_data = pd.DataFrame()
        synthetic_data = pd.DataFrame()
        real_validation_data = pd.DataFrame()
        metadata = {}
        prediction_column_name = 'prediction_column_name'
        minority_class_label = 'minority_class_label'
        classifier = 'XGBoost'
        fixed_precision_value = 0.8

        # Run
        BinaryClassifierRecallEfficacy.compute_breakdown(
            real_training_data=real_training_data,
            synthetic_data=synthetic_data,
            real_validation_data=real_validation_data,
            metadata=metadata,
            prediction_column_name=prediction_column_name,
            minority_class_label=minority_class_label,
            classifier=classifier,
            fixed_precision_value=fixed_precision_value,
        )

        # Assert
        mock_compute_breakdown.assert_called_once_with(
            real_training_data,
            synthetic_data,
            real_validation_data,
            metadata,
            prediction_column_name,
            minority_class_label,
            classifier,
            fixed_precision_value,
        )

    @patch('sdmetrics.single_table.data_augmentation.base.BaseDataAugmentationMetric.compute')
    def test_compute(self, mock_compute):
        """Test the compute method."""
        # Setup
        real_training_data = pd.DataFrame()
        synthetic_data = pd.DataFrame()
        real_validation_data = pd.DataFrame()
        metadata = {}
        prediction_column_name = 'prediction_column_name'
        minority_class_label = 'minority_class_label'
        classifier = 'XGBoost'
        fixed_precision_value = 0.8

        # Run
        BinaryClassifierRecallEfficacy.compute(
            real_training_data=real_training_data,
            synthetic_data=synthetic_data,
            real_validation_data=real_validation_data,
            metadata=metadata,
            prediction_column_name=prediction_column_name,
            minority_class_label=minority_class_label,
            classifier=classifier,
            fixed_precision_value=fixed_precision_value,
        )

        # Assert
        mock_compute.assert_called_once_with(
            real_training_data,
            synthetic_data,
            real_validation_data,
            metadata,
            prediction_column_name,
            minority_class_label,
            classifier,
            fixed_precision_value,
        )
