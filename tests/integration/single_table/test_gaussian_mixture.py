import numpy as np
import pandas as pd
import pytest

from sdmetrics.single_table.gaussian_mixture import GMLogLikelihood


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


def test_rank(ones, zeros, real_data, good_data, bad_data):
    worst = GMLogLikelihood.compute(ones, zeros)
    normalized_worst = GMLogLikelihood.normalize(worst)
    best = GMLogLikelihood.compute(ones, ones)
    normalized_best = GMLogLikelihood.normalize(best)

    assert GMLogLikelihood.min_value <= worst < best <= GMLogLikelihood.max_value
    assert 0.0 <= normalized_worst < normalized_best <= 1.0
