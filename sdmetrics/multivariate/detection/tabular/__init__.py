"""
This module implements machine learning methods for detecting synthetic
rows in a single table.
"""
from sdmetrics.multivariate.detection.tabular.base import TabularDetector
from sdmetrics.multivariate.detection.tabular.logistic import LogisticDetector
from sdmetrics.multivariate.detection.tabular.svc import SVCDetector

__all__ = ["TabularDetector", "LogisticDetector", "SVCDetector"]
