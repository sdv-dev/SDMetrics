import numpy as np
import pandas as pd
import pytest

from sdmetrics.single_table.efficacy.multiclass import (
    MulticlassDecisionTreeClassifier, MulticlassMLPClassifier)

METRICS = [
    MulticlassDecisionTreeClassifier,
    MulticlassMLPClassifier,
]


@pytest.fixture
def real_data():
    return pd.DataFrame({
        'a': np.random.normal(size=6000),
        'b': np.random.randint(0, 10, size=6000),
        'c': ['a', 'b', 'b', 'c', 'c', 'c'] * 1000,
        'd': [True, True, True, True, True, False] * 1000,
    })


@pytest.fixture
def good_data():
    return pd.DataFrame({
        'a': np.random.normal(loc=5, size=600),
        'b': np.random.randint(5, 15, size=600),
        'c': ['a', 'b', 'b', 'b', 'c', 'c'] * 100,
        'd': [True, True, True, True, False, False] * 100,
    })


@pytest.fixture
def bad_data():
    return pd.DataFrame({
        'a': np.random.normal(loc=10, scale=3, size=600),
        'b': np.random.randint(10, 20, size=600),
        'c': ['a', 'a', 'a', 'a', 'a', 'b'] * 100,
        'd': [True, False, False, False, False, False] * 100,
    })


@pytest.mark.parametrize('metric', METRICS)
def test_rank(metric, real_data, good_data, bad_data):
    bad = metric.compute(real_data, bad_data, target='c')
    good = metric.compute(real_data, good_data, target='c')
    real = metric.compute(real_data, real_data, target='c')

    assert metric.min_value <= bad
    assert bad < good
    assert good < real
    assert real <= metric.max_value
