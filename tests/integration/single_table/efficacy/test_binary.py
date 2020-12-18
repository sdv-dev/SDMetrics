import numpy as np
import pandas as pd
import pytest

from sdmetrics.single_table.efficacy.binary import (
    BinaryAdaBoostClassifier, BinaryDecisionTreeClassifier, BinaryLogisticRegression,
    BinaryMLPClassifier)

METRICS = [
    BinaryAdaBoostClassifier,
    BinaryDecisionTreeClassifier,
    BinaryLogisticRegression,
    BinaryMLPClassifier
]


def real_data():
    return pd.DataFrame({
        'a': np.random.normal(size=600),
        'b': np.random.randint(0, 10, size=600),
        'c': ['a', 'b', 'b', 'c', 'c', 'c'] * 100,
        'd': [True, True, True, True, True, False] * 100,
    })


def good_data():
    return pd.DataFrame({
        'a': np.random.normal(loc=0.01, size=600),
        'b': np.random.randint(0, 10, size=600),
        'c': ['a', 'b', 'b', 'b', 'c', 'c'] * 100,
        'd': [True, True, True, True, False, False] * 100,
    })


def bad_data():
    return pd.DataFrame({
        'a': np.random.normal(loc=5, scale=3, size=600),
        'b': np.random.randint(5, 15, size=600),
        'c': ['a', 'a', 'a', 'a', 'b', 'b'] * 100,
        'd': [True, False, False, False, False, False] * 100,
    })


@pytest.mark.parametrize('metric', METRICS)
def test_rank(metric):
    bad = metric.compute(real_data(), bad_data(), target='d')
    good = metric.compute(real_data(), good_data(), target='d')
    real = metric.compute(real_data(), real_data(), target='d')

    assert metric.min_value <= bad < good <= real <= metric.max_value
