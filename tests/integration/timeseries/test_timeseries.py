import pandas as pd
import pytest

from sdmetrics import compute_metrics
from sdmetrics.demos import load_timeseries_demo
from sdmetrics.timeseries.base import TimeSeriesMetric
from sdmetrics.timeseries.detection import LSTMDetection, TSFCDetection

METRICS = [
    LSTMDetection,
    TSFCDetection
]


@pytest.mark.parametrize('metric', METRICS)
def test_rank(metric):
    real_data, synthetic_data, metadata = load_timeseries_demo()

    real_score = metric.compute(real_data, real_data, metadata)
    synthetic_score = metric.compute(real_data, synthetic_data, metadata)

    assert metric.min_value <= synthetic_score <= real_score <= metric.max_value


def test_compute_all():
    real_data, synthetic_data, metadata = load_timeseries_demo()

    output = compute_metrics(
        TimeSeriesMetric.get_subclasses(),
        real_data,
        synthetic_data,
        metadata=metadata
    )

    assert not pd.isnull(output.score.mean())

    scores = output[output.score.notnull()]

    assert scores.score.between(scores.min_value, scores.max_value).all()
