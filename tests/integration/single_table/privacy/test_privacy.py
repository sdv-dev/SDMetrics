import numpy as np
import pandas as pd
import pytest

from sdmetrics.single_table.privacy.cap import *
from sdmetrics.single_table.privacy.nn import *
from sdmetrics.single_table.privacy.num_skl import *
from sdmetrics.single_table.privacy.base import CatPrivacyMetric, NumPrivacyMetric
from sdmetrics.single_table.privacy.cat_skl import CatKNNAttacker, CatNBAttacker, CatRFAttacker
from sdmetrics.single_table.privacy.ens import CatENS

Cat_metrics = CatPrivacyMetric.get_subclasses()
Num_metrics = NumPrivacyMetric.get_subclasses()

"""
TODO:
    1) Add categorical tests for only 1 column each
    2) Maybe add test for 1 key column 2 sensitive & vice versa
    3) Maybe add a few more categories? Although it really should work
    4) Do the same tests for the numerical
    5) Maybe some sanity checks, like if passed wrong key/sensitive columns
    6) unit test closest_neighbors (i think it doesnt work, and even if it did it's kinda
    questionable, if the value has never been seen in the synthetic data, hamming distance is inf)

"""

def cat_real_data():
    return pd.DataFrame({
        'key1': ['a', 'b', 'c', 'd', 'e'] * 20,
        'key2': [0, 1, 2, 3, 4] * 20,
        'sensitive1': ['a', 'b', 'c', 'd', 'e'] * 20,
        'sensitive2': [0, 1, 2, 3, 4] * 20
    })

def cat_perfect_synthetic_data():
    return pd.DataFrame({
        'key1': np.random.choice(['a', 'b', 'c', 'd', 'e'], 100),
        'key2': np.random.randint(0, 5, size=100),
        'sensitive1': np.random.choice(['f', 'g', 'h', 'i', 'j'], 100),
        'sensitive2': np.random.randint(5, 10, size=100)
    })

def cat_good_synthetic_data():
    return pd.DataFrame({
        'key1': np.random.choice(['a', 'b', 'c', 'd', 'e'], 100),
        'key2': np.random.randint(0, 5, size=100),
        'sensitive1': np.random.choice(['a', 'b', 'c', 'd', 'e'], 100),
        'sensitive2': np.random.randint(0, 5, size=100)
    })

def cat_bad_synthetic_data():
    return pd.DataFrame({
        'key1': ['a', 'b', 'c', 'd', 'e'] * 20,
        'key2': [0, 1, 2, 3, 4] * 20,
        'sensitive1': ['a', 'b', 'c', 'e', 'd'] * 20,
        'sensitive2': [0, 1, 2, 3, 4] * 20
    })



def num_data():
    return pd.DataFrame({
        'a': np.random.normal(size=50),
        'b': np.random.randint(0, 10, size=50),
        'c': np.random.normal(size=50),
        'd': np.random.normal(size=50)
    })

def data():
    return pd.DataFrame({
        'key': np.random.choice(['a', 'b', 'c', 'd', 'e'], 100),
        'sensitive': np.random.normal(size=100)
    })


@pytest.mark.parametrize('metric', Cat_metrics.values())
def test_cat(metric):
    if metric != CatENS:  # ENS is special since it requires additional args to work
        perfect = metric.compute(cat_real_data(), cat_perfect_synthetic_data(),
                key=['key1', 'key2'], sensitive=['sensitive1', 'sensitive2'])
        good = metric.compute(cat_real_data(), cat_good_synthetic_data(),
                key=['key1', 'key2'], sensitive=['sensitive1', 'sensitive2'])
        bad = metric.compute(cat_real_data(), cat_bad_synthetic_data(),
                key=['key1', 'key2'], sensitive=['sensitive1', 'sensitive2'])
        horrible = metric.compute(cat_real_data(), cat_real_data(),
                key=['key1', 'key2'], sensitive=['sensitive1', 'sensitive2'])
        x = metric.compute(data(), data(),
                key=['key', 'key2'], sensitive=['sensitive'])

        num = metric.compute(num_data(), num_data(), key=['a', 'b'], sensitive=['c', 'd'])
    else:
        model_kwargs = {'attackers': [CatNBAttacker, CatRFAttacker, CatKNNAttacker]}
        perfect = metric.compute(cat_real_data(), cat_perfect_synthetic_data(),
            key=['key1', 'key2'], sensitive=['sensitive1', 'sensitive2'], model_kwargs=model_kwargs)
        good = metric.compute(cat_real_data(), cat_good_synthetic_data(),
            key=['key1', 'key2'], sensitive=['sensitive1', 'sensitive2'], model_kwargs=model_kwargs)
        bad = metric.compute(cat_real_data(), cat_bad_synthetic_data(),
            key=['key1', 'key2'], sensitive=['sensitive1', 'sensitive2'], model_kwargs=model_kwargs)
        horrible = metric.compute(cat_real_data(), cat_real_data(),
            key=['key1', 'key2'], sensitive=['sensitive1', 'sensitive2'], model_kwargs=model_kwargs)
        x = metric.compute(data(), data(),
                key=['key', 'key2'], sensitive=['sensitive'], model_kwargs=model_kwargs)

        num = metric.compute(num_data(), num_data(), key=['a', 'b'], sensitive=['c', 'd'],
                             model_kwargs=model_kwargs)

    print(metric.min_value)
    print(perfect)
    print(good)
    print(bad)
    print(horrible)
    print(metric.max_value)
    print(x)
    assert 1 == 2
    assert metric.min_value <= perfect <= good <= bad <= horrible <= metric.max_value
    assert np.isnan(num)


@pytest.mark.parametrize('metric', Num_metrics.values())
def test_num(metric):
    #cat = metric.compute(cat_bad_data(), cat_bad_data(), key=['a', 'b'], sensitive=['c', 'd'])
    num = metric.compute(num_data(), num_data(), key=['a', 'b'], sensitive=['c', 'd'])
    #bad = metric.compute(bad_data(), bad_data(), key=['a', 'b'], sensitive=['c', 'd'])
    x = metric.compute(data(), data(),
                key=['key'], sensitive=['sensitive'])
    print("x:" + str(x))
    assert 1 == 2
    assert metric.min_value <= num <= metric.max_value
    #assert np.isnan(cat)
    #assert np.isnan(bad)
