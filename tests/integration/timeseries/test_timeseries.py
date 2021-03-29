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

    normalized_real_score = metric.normalize(real_score)
    normalized_synthetic_score = metric.normalize(synthetic_score)

    assert metric.min_value <= synthetic_score <= real_score <= metric.max_value
    assert 0.0 <= normalized_synthetic_score <= normalized_real_score <= 1.0


def test_compute_all():
    real_data, synthetic_data, metadata = load_timeseries_demo()

    output = compute_metrics(
        TimeSeriesMetric.get_subclasses(),
        real_data,
        synthetic_data,
        metadata=metadata
    )

    assert not pd.isnull(output.raw_score.mean())

    scores = output[output.raw_score.notnull()]

    assert scores.raw_score.between(scores.min_value, scores.max_value).all()

    scores = output[output.normalized_score.notnull()]

    assert scores.normalized_score.between(0.0, 1.0).all()
