import numpy as np
import pandas as pd
import pytest

from sdmetrics.single_table.privacy.base import CatPrivacyMetric, NumPrivacyMetric
from sdmetrics.single_table.privacy.cat_skl import CatKNNAttacker, CatNBAttacker,\
    CatRFAttacker
from sdmetrics.single_table.privacy.ens import CatENS

Cat_metrics = CatPrivacyMetric.get_subclasses()
Num_metrics = NumPrivacyMetric.get_subclasses()


def cat_data():
    return pd.DataFrame({
        'a': np.random.choice(['a', 'b', 'c', 'd', 'e'], 50),
        'b': np.random.randint(0, 10, size=50),
        'c': np.random.choice(['a', 'b', 'c', 'd', 'e'], 50),
        'd': np.random.choice([True, False], 50)
    })


def num_data():
    return pd.DataFrame({
        'a': np.random.normal(size=50),
        'b': np.random.randint(0, 10, size=50),
        'c': np.random.normal(size=50),
        'd': np.random.normal(size=50)
    })


def bad_data():
    return pd.DataFrame({
        'a': np.random.normal(size=50),
        'b': np.random.randint(0, 10, size=50),
        'c': np.random.choice(['a', 'b', 'c', 'd', 'e'], 50),
        'd': np.random.choice([True, False], 50),
    })


@pytest.mark.parametrize('metric', Cat_metrics.values())
def test_cat(metric):
    if metric != CatENS:  # ENS is special since it requires additional args to work
        cat = metric.compute(cat_data(), cat_data(), key=['a', 'b'], sensitive=['c', 'd'])
        num = metric.compute(num_data(), num_data(), key=['a', 'b'], sensitive=['c', 'd'])
        bad = metric.compute(bad_data(), bad_data(), key=['a', 'b'], sensitive=['c', 'd'])
    else:
        model_kwargs = {'attackers': [CatNBAttacker, CatRFAttacker, CatKNNAttacker]}
        cat = metric.compute(cat_data(), cat_data(), key=['a', 'b'], sensitive=['c', 'd'],
                             model_kwargs=model_kwargs)
        num = metric.compute(num_data(), num_data(), key=['a', 'b'], sensitive=['c', 'd'],
                             model_kwargs=model_kwargs)
        bad = metric.compute(bad_data(), bad_data(), key=['a', 'b'], sensitive=['c', 'd'],
                             model_kwargs=model_kwargs)
    assert metric.min_value <= cat <= metric.max_value
    assert np.isnan(num)
    assert np.isnan(bad)


@pytest.mark.parametrize('metric', Num_metrics.values())
def test_num(metric):
    cat = metric.compute(cat_data(), cat_data(), key=['a', 'b'], sensitive=['c', 'd'])
    num = metric.compute(num_data(), num_data(), key=['a', 'b'], sensitive=['c', 'd'])
    bad = metric.compute(bad_data(), bad_data(), key=['a', 'b'], sensitive=['c', 'd'])

    assert metric.min_value <= num <= metric.max_value
    assert np.isnan(cat)
    assert np.isnan(bad)
