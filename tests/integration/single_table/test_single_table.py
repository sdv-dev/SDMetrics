import numpy as np
import pandas as pd
import pytest

from sdmetrics import compute_metrics
from sdmetrics.demos import load_single_table_demo
from sdmetrics.single_table.base import SingleTableMetric
from sdmetrics.single_table.bayesian_network import BNLikelihood, BNLogLikelihood
from sdmetrics.single_table.detection import LogisticDetection, SVCDetection
from sdmetrics.single_table.gaussian_mixture import GMLogLikelihood
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
    GMLogLikelihood,
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
    best = metric.compute(ones, ones)

    bad = metric.compute(real_data, bad_data)
    good = metric.compute(real_data, good_data)
    real = metric.compute(real_data, real_data)

    assert metric.min_value <= worst < best <= metric.max_value
    assert metric.min_value <= bad < good < real <= metric.max_value


def test_compute_all():
    real_data, synthetic_data, metadata = load_single_table_demo()

    output = compute_metrics(
        SingleTableMetric.get_subclasses(),
        real_data,
        synthetic_data,
        metadata=metadata
    )

    assert not pd.isnull(output.score.mean())

    scores = output[output.score.notnull()]

    assert scores.score.between(scores.min_value, scores.max_value).all()
