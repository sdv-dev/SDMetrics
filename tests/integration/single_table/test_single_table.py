import numpy as np
import pandas as pd
import pytest

from sdmetrics import compute_metrics
from sdmetrics.demos import load_single_table_demo
from sdmetrics.goal import Goal
from sdmetrics.single_table.base import SingleTableMetric
from sdmetrics.single_table.bayesian_network import BNLikelihood, BNLogLikelihood
from sdmetrics.single_table.detection import LogisticDetection, SVCDetection
from sdmetrics.single_table.multi_column_pairs import ContinuousKLDivergence, DiscreteKLDivergence
from sdmetrics.single_table.multi_single_column import CSTest, KSTest, KSTestExtended

METRICS = [
    CSTest,
    KSTest,
    KSTestExtended,
    LogisticDetection,
    SVCDetection,
    ContinuousKLDivergence,
    DiscreteKLDivergence,
    BNLikelihood,
    BNLogLikelihood,
]


@pytest.fixture
def ones():
    return pd.DataFrame({
        'a': [1] * 300,
        'b': [True] * 300,
        'c': [1.0] * 300,
        'd': [True] * 300,
    })


@pytest.fixture
def zeros():
    return pd.DataFrame({
        'a': [0] * 300,
        'b': [False] * 300,
        'c': [0.0] * 300,
        'd': [False] * 300,
    })


@pytest.fixture
def real_data():
    return pd.DataFrame({
        'a': np.random.normal(size=1800),
        'b': np.random.randint(0, 10, size=1800),
        'c': ['a', 'b', 'b', 'c', 'c', 'c'] * 300,
        'd': [True, True, True, True, True, False] * 300,
    })


@pytest.fixture
def good_data():
    return pd.DataFrame({
        'a': np.random.normal(loc=0.01, size=1800),
        'b': np.random.randint(0, 10, size=1800),
        'c': ['a', 'b', 'b', 'b', 'c', 'c'] * 300,
        'd': [True, True, True, True, False, False] * 300,
    })


@pytest.fixture
def bad_data():
    return pd.DataFrame({
        'a': np.random.normal(loc=5, scale=3, size=1800),
        'b': np.random.randint(5, 15, size=1800),
        'c': ['a', 'a', 'a', 'a', 'b', 'b'] * 300,
        'd': [True, False, False, False, False, False] * 300,
    })


@pytest.mark.parametrize('metric', METRICS)
def test_rank(metric, ones, zeros, real_data, good_data, bad_data):
    worst = metric.compute(ones, zeros)
    normalized_worst = metric.normalize(worst)
    best = metric.compute(ones, ones)
    normalized_best = metric.normalize(best)

    bad = metric.compute(real_data, bad_data)
    normalized_bad = metric.normalize(bad)
    good = metric.compute(real_data, good_data)
    normalized_good = metric.normalize(good)
    real = metric.compute(real_data, real_data)
    normalized_real = metric.normalize(real)

    if metric.goal == Goal.MAXIMIZE:
        assert metric.min_value <= worst < best <= metric.max_value
        assert metric.min_value <= bad < good < real <= metric.max_value
    else:
        assert metric.min_value <= best < worst <= metric.max_value
        assert metric.min_value <= real < good < bad <= metric.max_value

    assert 0.0 <= normalized_worst < normalized_best <= 1.0
    assert 0.0 <= normalized_bad < normalized_good < normalized_real <= 1.0


def test_compute_all():
    real_data, synthetic_data, metadata = load_single_table_demo()

    output = compute_metrics(
        SingleTableMetric.get_subclasses(),
        real_data,
        synthetic_data,
        metadata=metadata
    )

    assert not pd.isnull(output.raw_score.mean())

    scores = output[output.raw_score.notnull()]
    assert scores.raw_score.between(scores.min_value, scores.max_value).all()

    scores = output[output.normalized_score.notnull()]
    assert scores.normalized_score.between(0.0, 1.0).all()
