"""Binary classifier recall efficacy metric."""

from sdmetrics.single_table.data_augmentation.base import BaseDataAugmentationMetric


class BinaryClassifierRecallEfficacy(BaseDataAugmentationMetric):
    """Binary classifier recall efficacy metric."""

    name = 'Binary Classifier Recall Efficacy'
    metric_name = 'recall'
