"""
This module implements machine learning methods for detecting synthetic
rows in a single table.
"""
from sdmetrics.detection.tabular.base import TabularDetector
from sdmetrics.detection.tabular.sklearn import LogisticDetector, SVCDetector

__all__ = ["TabularDetector", "LogisticDetector", "SVCDetector"]
