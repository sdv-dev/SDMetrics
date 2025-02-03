"""Test for the Binary Classifier Precision Efficacy metrics."""

from sdmetrics.single_table.data_augmentation.binary_classifier_precision_efficacy import (
    BinaryClassifierPrecisionEfficacy,
)


class TestBinaryClassifierPrecisionEfficacy:
    def test_class_attributes(self):
        """Test the class attributes."""
        # Setup
        expected_name = 'Binary Classifier Precision Efficacy'
        expected_metric_name = 'precision'

        # Run and Assert
        assert BinaryClassifierPrecisionEfficacy.name == expected_name
        assert BinaryClassifierPrecisionEfficacy.metric_name == expected_metric_name
