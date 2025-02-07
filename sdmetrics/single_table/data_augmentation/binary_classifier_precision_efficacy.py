"""Binary classifier precision efficacy metric."""

from sdmetrics.single_table.data_augmentation.base import BaseDataAugmentationMetric


class BinaryClassifierPrecisionEfficacy(BaseDataAugmentationMetric):
    """Binary classifier precision efficacy metric."""

    name = 'Binary Classifier Precision Efficacy'
    metric_name = 'precision'
