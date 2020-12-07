import numpy as np
import pandas as pd
import pytest

from sdmetrics.single_table.bayesian_network import BNLikelihood, BNLogLikelihood
from sdmetrics.single_table.detection import LogisticDetection, SVCDetection
from sdmetrics.single_table.gaussian_mixture import GMLikelihood
from sdmetrics.single_table.multi_column_pairs import ContinuousKLDivergence, DiscreteKLDivergence
from sdmetrics.single_table.multi_single_column import CSTest, KSTest

METRICS = [
    CSTest,
    KSTest,
    LogisticDetection,
    SVCDetection,
    ContinuousKLDivergence,
    DiscreteKLDivergence,
    BNLikelihood,
    BNLogLikelihood,
    GMLikelihood,
]


@pytest.fixture
def ones():
    return pd.DataFrame({
        'a': [1] * 100,
        'b': [True] * 100,
        'c': [1.0] * 100,
        'd': [True] * 100,
    })


@pytest.fixture
def zeros():
    return pd.DataFrame({
        'a': [0] * 100,
        'b': [False] * 100,
        'c': [0.0] * 100,
        'd': [False] * 100,
    })


@pytest.fixture
def real_data():
    return pd.DataFrame({
        'a': np.random.normal(size=600),
        'b': np.random.randint(0, 10, size=600),
        'c': ['a', 'b', 'b', 'c', 'c', 'c'] * 100,
        'd': [True, True, True, True, True, False] * 100,
    })


@pytest.fixture
def good_data():
    return pd.DataFrame({
        'a': np.random.normal(loc=0.01, size=600),
        'b': np.random.randint(0, 10, size=600),
        'c': ['a', 'b', 'b', 'b', 'c', 'c'] * 100,
        'd': [True, True, True, True, False, False] * 100,
    })


@pytest.fixture
def bad_data():
    return pd.DataFrame({
        'a': np.random.normal(loc=5, scale=3, size=600),
        'b': np.random.randint(5, 15, size=600),
        'c': ['a', 'a', 'a', 'a', 'b', 'b'] * 100,
        'd': [True, False, False, False, False, False] * 100,
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
