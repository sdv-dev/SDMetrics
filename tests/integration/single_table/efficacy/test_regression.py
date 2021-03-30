import sys

import numpy as np
import pandas as pd
import pytest

from sdmetrics.single_table.efficacy.regression import LinearRegression, MLPRegressor

METRICS = [
    LinearRegression,
    MLPRegressor,
]


# if sys.version_info.minor == 6:
#     np.random.seed(42000)


@pytest.fixture
def real_data():
    return pd.DataFrame({
        'a': np.random.normal(size=1000),
        'b': np.random.randint(0, 10, size=1000),
        'c': ['a', 'b', 'b', 'c'] * 250,
        'd': [True, True, True, False] * 250,
    })


@pytest.fixture
def good_data():
    return pd.DataFrame({
        'a': np.random.normal(loc=0.01, size=1000),
        'b': np.random.randint(0, 10, size=1000),
        'c': ['a', 'b', 'c', 'c'] * 250,
        'd': [True, True, False, False] * 250,
    })


@pytest.fixture
def bad_data():
    return pd.DataFrame({
        'a': np.random.normal(loc=5, scale=3, size=1000),
        'b': np.random.randint(5, 15, size=1000),
        'c': ['a', 'a', 'a', 'b'] * 250,
        'd': [True, False, False, False] * 250,
    })


@pytest.mark.parametrize('metric', METRICS)
def test_rank(metric, real_data, good_data, bad_data):
    bad = metric.compute(real_data, bad_data, target='a')
    good = metric.compute(real_data, good_data, target='a')
    real = metric.compute(real_data, real_data, target='a')

    normalized_bad = metric.normalize(bad)
    normalized_good = metric.normalize(good)
    normalized_real = metric.normalize(real)

    assert metric.min_value <= bad < good < real <= metric.max_value
    assert 0.0 <= normalized_bad <= normalized_good <= normalized_real <= 1.0
