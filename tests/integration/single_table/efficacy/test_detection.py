import pytest

from sdmetrics import load_demo
from sdmetrics.single_table.detection import LogisticDetection, SVCDetection

METRICS = [
    LogisticDetection,
    SVCDetection
]


@pytest.mark.parametrize('metric', METRICS)
def test_primary_key(metric):
    """Test that primary keys don't affect detection metric."""
    real_data_with_primary_key, synthetic_data_with_primary_key, metadata = load_demo(
        modality='single_table')

    real_data_sin_primary_key = real_data_with_primary_key.drop(metadata['primary_key'], axis=1)
    synthetic_data_sin_primary_key = synthetic_data_with_primary_key.drop(
        metadata['primary_key'], axis=1)

    test_with_primary_key = metric.compute(real_data_with_primary_key,
                                           synthetic_data_with_primary_key, metadata)
    test_sin_primary_key = metric.compute(real_data_sin_primary_key,
                                          synthetic_data_sin_primary_key)

    normalized_with_primary_key = metric.normalize(test_with_primary_key)
    normalized_sin_primary_key = metric.normalize(test_sin_primary_key)

    # Approximately equal because detection metrics vary when receiving the same data.
    assert pytest.approx(normalized_with_primary_key, abs=0.06) == normalized_sin_primary_key
