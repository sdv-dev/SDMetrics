# -*- coding: utf-8 -*-

"""Top-level package for SDMetrics."""

__author__ = 'MIT Data To AI Lab'
__email__ = 'dailabmit@gmail.com'
__version__ = '0.1.0.dev2'

import pandas as pd

from sdmetrics import (
    column_pairs, demos, goal, multi_table, single_column, single_table, timeseries)
from sdmetrics.demos import load_demo

__all__ = [
    'demos',
    'load_demo',
    'goal',
    'multi_table',
    'column_pairs',
    'single_column',
    'single_table',
    'timeseries',
]


def compute_metrics(metrics, real_data, synthetic_data, metadata=None, **kwargs):
    """Compute a collection of metrics on the given data.

    Args:
        metrics (list[sdmetrics.base.BaseMetric]):
            Metrics to compute.
        real_data:
            Data from the real dataset
        synthetic_data:
            Data from the synthetic dataset
        metadata (dict):
            Dataset metadata.
        **kwargs:
            Any additional arguments to pass to the metrics.

    Returns:
        pandas.DataFrame:
            Dataframe containing the metric scores, as well as information
            about each metric such as the min and max values and its goal.
    """
    # Only add metadata to kwargs if passed, to stay compatible
    # with metrics that do not expect a metadata argument
    if metadata is not None:
        kwargs['metadata'] = metadata

    scores = []
    for name, metric in metrics.items():
        try:
            score = metric.compute(real_data, synthetic_data, **kwargs)
        except Exception:
            score = None

        scores.append({
            'metric': name,
            'name': metric.name,
            'score': score,
            'min_value': metric.min_value,
            'max_value': metric.max_value,
            'goal': metric.goal.name,
        })

    return pd.DataFrame(scores)
