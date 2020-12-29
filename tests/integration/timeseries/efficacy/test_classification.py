import pytest

from sdmetrics.demos import load_timeseries_demo
from sdmetrics.timeseries.efficacy.classification import (
    LSTMClassifierEfficacy, TSFClassifierEfficacy)

METRICS = [
    LSTMClassifierEfficacy,
    TSFClassifierEfficacy,
]


@pytest.mark.parametrize('metric', METRICS)
def test_rank(metric):
    real_data, synthetic_data, metadata = load_timeseries_demo()

    real_score = metric.compute(real_data, real_data, metadata, target='region')
    synthetic_score = metric.compute(real_data, synthetic_data, metadata, target='region')

    assert metric.min_value <= synthetic_score <= real_score <= metric.max_value
