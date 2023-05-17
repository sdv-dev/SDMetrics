"""Machine Learning Detection metrics for single table datasets."""

from sdmetrics.single_table.detection.sklearn import (
    GradientBoostingDetection, LogisticDetection, SVCDetection)

__all__ = [
    'GradientBoostingDetection',
    'LogisticDetection',
    'SVCDetection'
]
