"""Test for the Binary Classifier Precision Efficacy metrics."""

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
