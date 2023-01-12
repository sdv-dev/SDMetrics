import numpy as np
import pandas as pd
import pytest
from sklearn.datasets import load_diabetes

from sdmetrics.single_table.efficacy.regression import LinearRegression, MLPRegressor

METRICS = [
    LinearRegression,
    MLPRegressor,
]


@pytest.fixture()
def test_data():
    boston = load_diabetes()
    data = pd.DataFrame(boston.data, columns=boston.feature_names)
    data['target'] = boston.target
    return data


@pytest.fixture()
def good_data():
    boston = load_diabetes()
    data = pd.DataFrame(boston.data, columns=boston.feature_names)

    columns = len(data.columns)
    rows = len(data)
    data = boston.data

    stds = data.std(axis=0) / 4
    zeros = np.zeros(columns)
    noise = np.random.normal(loc=zeros, scale=stds, size=(rows, columns))
    good = data + noise * 4

    good = pd.DataFrame(good, columns=boston.feature_names)
    good['target'] = boston.target
    return good


@pytest.fixture()
def bad_data():
    boston = load_diabetes()
    data = pd.DataFrame(boston.data, columns=boston.feature_names)

    stds = data.std(axis=0)
    mus = data.mean(axis=0)
    columns = len(data.columns)
    rows = len(data)
    bad = np.random.normal(loc=mus, scale=stds, size=(rows, columns))
    bad = pd.DataFrame(bad, columns=data.columns)

    bad['target'] = boston.target

    return bad


@pytest.mark.parametrize('metric', METRICS)
def test_rank(metric, test_data, good_data, bad_data):
    bad = metric.compute(test_data, bad_data, target='target')
    good = metric.compute(test_data, good_data, target='target')
    test = metric.compute(test_data, test_data, target='target')

    normalized_bad = metric.normalize(bad)
    normalized_good = metric.normalize(good)
    normalized_test = metric.normalize(test)

    assert metric.min_value <= bad < good < test <= metric.max_value
    assert 0.0 <= normalized_bad <= normalized_good <= normalized_test <= 1.0
