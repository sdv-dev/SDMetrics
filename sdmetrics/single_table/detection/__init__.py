"""Machine Learning Detection metrics for single table datasets."""

from sdmetrics.single_table.detection.sklearn import LogisticDetection, SVCDetection, GradientBoostingDetection

__all__ = [
    'LogisticDetection',
    'SVCDetection',
    'GradientBoostingDetection'
]
