import pandas as pd

from sdmetrics import compute_metrics
from sdmetrics.demos import load_multi_table_demo
from sdmetrics.multi_table.base import MultiTableMetric


def test_compute_all():
    real_data, synthetic_data, metadata = load_multi_table_demo()

    output = compute_metrics(
        MultiTableMetric.get_subclasses(),
        real_data,
        synthetic_data,
        metadata=metadata
    )

    assert not pd.isnull(output.raw_score.mean())

    scores = output[output.raw_score.notnull()]

    assert scores.raw_score.between(scores.min_value, scores.max_value).all()
