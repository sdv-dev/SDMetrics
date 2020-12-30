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


def real_data(as_str=False):
    data = pd.DataFrame({
        'a': np.random.normal(size=600),
        'b': np.random.randint(0, 10, size=600),
        'c': ['a', 'b', 'b', 'c', 'c', 'c'] * 100,
        'd': [True, True, True, True, True, False] * 100,
    })
    if as_str:
        data['d'] = data['d'].astype(str)

    return data


def good_data(as_str=False):
    data = pd.DataFrame({
        'a': np.random.normal(loc=0.01, size=600),
        'b': np.random.randint(0, 10, size=600),
        'c': ['a', 'b', 'b', 'b', 'c', 'c'] * 100,
        'd': [True, True, True, True, False, False] * 100,
    })
    if as_str:
        data['d'] = data['d'].astype(str)

    return data


def bad_data(as_str=False):
    data = pd.DataFrame({
        'a': np.random.normal(loc=5, scale=3, size=600),
        'b': np.random.randint(5, 15, size=600),
        'c': ['a', 'a', 'a', 'a', 'b', 'b'] * 100,
        'd': [True, False, False, False, False, False] * 100,
    })
    if as_str:
        data['d'] = data['d'].astype(str)

    return data


@pytest.mark.parametrize('metric', METRICS)
def test_rank(metric):
    bad = metric.compute(real_data(), bad_data(), target='d')
    good = metric.compute(real_data(), good_data(), target='d')
    real = metric.compute(real_data(), real_data(), target='d')

    assert metric.min_value <= bad < good <= real <= metric.max_value


@pytest.mark.parametrize('metric', METRICS)
def test_rank_object(metric):
    bad = metric.compute(real_data(True), bad_data(True), target='d')
    good = metric.compute(real_data(True), good_data(True), target='d')
    real = metric.compute(real_data(True), real_data(True), target='d')

    assert metric.min_value <= bad < good <= real <= metric.max_value
