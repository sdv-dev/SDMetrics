import numpy as np
import pandas as pd
import pytest

from sdmetrics.multi_table.detection.parent_child import (
    LogisticParentChildDetection, SVCParentChildDetection)

METRICS = [LogisticParentChildDetection, SVCParentChildDetection]


def ones():
    parent = pd.DataFrame({
        'id': range(10),
        'a': [1] * 10,
        'b': [True] * 10,
    })
    child = pd.DataFrame({
        'parent_id': list(range(10)) * 10,
        'a': [1] * 100,
        'b': [True] * 100,
    })
    return {'parent': parent, 'child': child}


def zeros():
    parent = pd.DataFrame({
        'id': range(10),
        'a': [0] * 10,
        'b': [False] * 10,
    })
    child = pd.DataFrame({
        'parent_id': list(range(10)) * 10,
        'a': [0] * 100,
        'b': [False] * 100,
    })
    return {'parent': parent, 'child': child}


def real_data():
    parent = pd.DataFrame({
        'id': range(60),
        'a': np.random.normal(size=60),
        'b': np.random.randint(0, 10, size=60),
        'c': ['a', 'b', 'b', 'c', 'c', 'c'] * 10,
        'd': [True, True, True, True, True, False] * 10,
    })
    child = pd.DataFrame({
        'parent_id': list(range(60)) * 10,
        'a': np.random.normal(size=600),
        'b': np.random.randint(0, 10, size=600),
        'c': ['a', 'b', 'b', 'c', 'c', 'c'] * 100,
        'd': [True, True, True, True, True, False] * 100,
    })
    return {'parent': parent, 'child': child}


def good_data():
    parent = pd.DataFrame({
        'id': range(60),
        'a': np.random.normal(loc=0.01, size=60),
        'b': np.random.randint(1, 11, size=60),
        'c': ['a', 'b', 'b', 'c', 'c', 'c'] * 10,
        'd': [True, True, True, True, True, False] * 10,
    })
    child = pd.DataFrame({
        'parent_id': list(range(60)) * 10,
        'a': np.random.normal(loc=0.01, size=600),
        'b': np.random.randint(1, 11, size=600),
        'c': ['a', 'b', 'b', 'c', 'c', 'c'] * 100,
        'd': [True, True, True, True, True, False] * 100,
    })
    return {'parent': parent, 'child': child}


def bad_data():
    parent = pd.DataFrame({
        'id': range(60),
        'a': np.random.normal(loc=5, scale=3, size=60),
        'b': np.random.randint(5, 15, size=60),
        'c': ['a', 'a', 'a', 'a', 'b', 'b'] * 10,
        'd': [True, False, False, False, False, False] * 10,
    })
    child = pd.DataFrame({
        'parent_id': list(range(60)) * 10,
        'a': np.random.normal(loc=5, scale=3, size=600),
        'b': np.random.randint(5, 15, size=600),
        'c': ['a', 'a', 'a', 'a', 'b', 'b'] * 100,
        'd': [True, False, False, False, False, False] * 100,
    })
    return {'parent': parent, 'child': child}


FKS = [
    ('parent', 'id', 'child', 'parent_id')
]


@pytest.mark.parametrize('metric', METRICS)
def test_max(metric):
    output = metric.compute(ones(), ones(), foreign_keys=FKS)
    normalized = metric.normalize(output)

    assert output == 1
    assert normalized == 1


@pytest.mark.parametrize('metric', METRICS)
def test_min(metric):
    output = metric.compute(ones(), zeros(), foreign_keys=FKS)
    normalized = metric.normalize(output)

    assert np.round(output, decimals=5) == 0
    assert np.round(normalized, decimals=5) == 0


@pytest.mark.parametrize('metric', METRICS)
def test_good(metric):
    output = metric.compute(real_data(), good_data(), foreign_keys=FKS)
    normalized = metric.normalize(output)

    assert 0.5 < output <= 1
    assert 0.5 < normalized <= 1


@pytest.mark.parametrize('metric', METRICS)
def test_bad(metric):
    output = metric.compute(real_data(), bad_data(), foreign_keys=FKS)
    normalized = metric.normalize(output)

    assert 0 <= output < 0.5
    assert 0 <= normalized < 0.5
