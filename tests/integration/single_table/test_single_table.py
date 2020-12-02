import numpy as np
import pandas as pd
import pytest

from sdmetrics.single_table.detection import LogisticDetection, SVCDetection
from sdmetrics.single_table.single_column import CSTest, KSTest

METRICS = [CSTest, KSTest, LogisticDetection, SVCDetection]


@pytest.fixture
def ones():
    return pd.DataFrame({
        'a': [1] * 100,
        'b': [True] * 100,
    })


@pytest.fixture
def zeros():
    return pd.DataFrame({
        'a': [0] * 100,
        'b': [False] * 100,
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
def test_max(metric, ones):
    output = metric.compute(ones, ones.copy())

    assert output == 1


@pytest.mark.parametrize('metric', METRICS)
def test_min(metric, ones, zeros):
    output = metric.compute(ones, zeros)

    assert np.round(output, decimals=5) == 0


@pytest.mark.parametrize('metric', METRICS)
def test_good(metric, real_data, good_data):
    output = metric.compute(real_data, good_data)

    assert 0.5 < output <= 1


@pytest.mark.parametrize('metric', METRICS)
def test_bad(metric, real_data, bad_data):
    output = metric.compute(real_data, bad_data)

    assert 0 <= output < 0.5
