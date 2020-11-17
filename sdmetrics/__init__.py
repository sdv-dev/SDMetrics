# -*- coding: utf-8 -*-

"""Top-level package for SDMetrics."""

__author__ = 'MIT Data To AI Lab'
__email__ = 'dailabmit@gmail.com'
__version__ = '0.0.3.dev0'

from sdmetrics.evaluation import evaluate
from sdmetrics.multivariate import constraint, detection, efficacy, statistical

__all__ = [
    'constraint',
    'detection',
    'efficacy',
    'evaluate',
    'statistical',
]
