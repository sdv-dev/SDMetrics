import pandas as pd
import pytest

from sdmetrics import compute_metrics
from sdmetrics.demos import load_timeseries_demo
from sdmetrics.timeseries.base import TimeSeriesMetric
from sdmetrics.timeseries.detection import LSTMDetection

METRICS = [
    LSTMDetection,
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

    assert not pd.isna(output.raw_score.mean())

    scores = output[output.raw_score.notna()]

    assert scores.raw_score.between(scores.min_value, scores.max_value).all()

    scores = output[output.normalized_score.notna()]

    assert scores.normalized_score.between(0.0, 1.0).all()


def test_compute_lstmdetection_multiple_categorical_columns():
    """Test LSTMDetection metric handles multiple categorical columns."""
    # Setup
    real_data, synthetic_data, metadata = load_timeseries_demo()
    metadata['columns']['day_of_week'] = {'sdtype': 'categorical'}
    day_map = {
        0: 'Sun', 1: 'Mon', 2: 'Tues', 3: 'Wed', 4: 'Thurs', 5: 'Fri', 6: 'Sat'
    }
    real_data['day_of_week'] = real_data['day_of_week'].replace(day_map)
    synthetic_data['day_of_week'] = synthetic_data['day_of_week'].clip(0, 6).replace(day_map)

    # Run
    output = LSTMDetection.compute(
        real_data,
        synthetic_data,
        metadata=metadata
    )

    # Assert
    assert not pd.isna(output)
    assert LSTMDetection.min_value <= output <= LSTMDetection.max_value
