"""
This module implements machine learning methods for detecting synthetic
rows in a single table.
"""
from .base import TabularDetector
from .logistic import LogisticDetector
from .svc import SVCDetector

__all__ = ["TabularDetector", "LogisticDetector", "SVCDetector"]
