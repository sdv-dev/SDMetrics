import pytest

from sdmetrics.timeseries.efficacy.classification import LSTMClassifierEfficacy

METRICS = [
    LSTMClassifierEfficacy,
]


@pytest.mark.parametrize('metric', METRICS)
def test_rank(metric, converted_datetime_timeseries_demo):
    real_data, synthetic_data, metadata = converted_datetime_timeseries_demo

    real_score = metric.compute(real_data, real_data, metadata, target='region')
    synthetic_score = metric.compute(real_data, synthetic_data, metadata, target='region')

    normalized_real_score = metric.normalize(real_score)
    normalized_synthetic_score = metric.normalize(synthetic_score)

    assert metric.min_value <= synthetic_score <= real_score <= metric.max_value
    assert 0.0 <= normalized_synthetic_score <= normalized_real_score <= 1.0
