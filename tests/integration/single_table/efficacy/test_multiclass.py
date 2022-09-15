import numpy as np
import pandas as pd
import pytest
from sklearn.datasets import load_wine

from sdmetrics.single_table.efficacy.multiclass import (
    MulticlassDecisionTreeClassifier, MulticlassMLPClassifier)

METRICS = [
    MulticlassDecisionTreeClassifier,
    MulticlassMLPClassifier,
]


@pytest.fixture()
def test_data():
    return load_wine(as_frame=True).frame


@pytest.fixture()
def good_data():
    wine = load_wine(as_frame=True)
    data = wine.data
    stds = data.std(axis=0) * 2.5
    columns = len(data.columns)
    rows = len(data)
    zeros = np.zeros(columns)
    noise = np.random.normal(loc=zeros, scale=stds, size=(rows, columns))
    good = data + noise
    good['target'] = wine.target
    return good


@pytest.fixture()
def bad_data():
    wine = load_wine(as_frame=True)
    data = wine.data
    stds = data.std(axis=0)
    mus = data.mean(axis=0)
    columns = len(data.columns)
    rows = len(data)
    bad = np.random.normal(loc=mus, scale=stds, size=(rows, columns))
    bad = pd.DataFrame(bad, columns=data.columns)
    bad['target'] = wine.target

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
    assert 0.0 <= normalized_bad < normalized_good <= normalized_test <= 1.0
