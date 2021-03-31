import numpy as np
import pandas as pd
import pytest

from sdmetrics.multi_table.multi_single_table import (
    CSTest, KSTest, LogisticDetection, SVCDetection)

METRICS = [CSTest, KSTest, LogisticDetection, SVCDetection]


@pytest.fixture
def ones():
    data = pd.DataFrame({
        'a': [1] * 100,
        'b': [True] * 100,
    })
    return {'a': data, 'b': data.copy()}


@pytest.fixture
def zeros():
    data = pd.DataFrame({
        'a': [0] * 100,
        'b': [False] * 100,
    })
    return {'a': data, 'b': data.copy()}


@pytest.fixture
def real_data():
    data = pd.DataFrame({
        'a': np.random.normal(size=600),
        'b': np.random.randint(0, 10, size=600),
        'c': ['a', 'b', 'b', 'c', 'c', 'c'] * 100,
        'd': [True, True, True, True, True, False] * 100,
    })
    return {'a': data, 'b': data.copy()}


@pytest.fixture
def good_data():
    data = pd.DataFrame({
        'a': np.random.normal(loc=0.01, size=600),
        'b': np.random.randint(0, 10, size=600),
        'c': ['a', 'b', 'b', 'b', 'c', 'c'] * 100,
        'd': [True, True, True, True, False, False] * 100,
    })
    return {'a': data, 'b': data.copy()}


@pytest.fixture
def bad_data():
    data = pd.DataFrame({
        'a': np.random.normal(loc=5, scale=3, size=600),
        'b': np.random.randint(5, 15, size=600),
        'c': ['a', 'a', 'a', 'a', 'b', 'b'] * 100,
        'd': [True, False, False, False, False, False] * 100,
    })
    return {'a': data, 'b': data.copy()}


@pytest.mark.parametrize('metric', METRICS)
def test_max(metric, ones):
    output = metric.compute(ones, ones.copy())
    normalized = metric.normalize(output)

    assert output == 1
    assert normalized == 1


@pytest.mark.parametrize('metric', METRICS)
def test_min(metric, ones, zeros):
    output = metric.compute(ones, zeros)
    normalized = metric.normalize(output)

    assert np.round(output, decimals=5) == 0
    assert np.round(normalized, decimals=5) == 0


@pytest.mark.parametrize('metric', METRICS)
def test_good(metric, real_data, good_data):
    output = metric.compute(real_data, good_data)
    normalized = metric.normalize(output)

    assert 0.5 < output <= 1
    assert 0.5 < normalized <= 1


@pytest.mark.parametrize('metric', METRICS)
def test_bad(metric, real_data, bad_data):
    output = metric.compute(real_data, bad_data)
    normalized = metric.normalize(output)

    assert 0 <= output < 0.5
    assert 0 <= normalized < 0.5


@pytest.mark.parametrize('metric', METRICS)
def test_fail(metric):
    with pytest.raises(ValueError):
        metric.compute({'a': None, 'b': None}, {'a': None})
