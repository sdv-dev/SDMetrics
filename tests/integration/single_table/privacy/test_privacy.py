import numpy as np
import pandas as pd
import pytest

from sdmetrics.single_table.privacy.cap import *
from sdmetrics.single_table.privacy.nn import *
from sdmetrics.single_table.privacy.num_skl import *
from sdmetrics.single_table.privacy.base import CategoricalPrivacyMetric, NumPrivacyMetric
from sdmetrics.single_table.privacy.cat_skl import CategoricalKNNAttacker, CategoricalNBAttacker, CategoricalRFAttacker
from sdmetrics.single_table.privacy.ens import CategoricalENS

Cat_metrics = CategoricalPrivacyMetric.get_subclasses()
Num_metrics = NumPrivacyMetric.get_subclasses()

"""
TODO:
    5) Maybe some sanity checks, like if passed wrong key/sensitive columns
    6) unit test closest_neighbors (i think it doesnt work, and even if it did it's kinda
    questionable, if the value has never been seen in the synthetic data, hamming distance is inf)
    7) Line 14 sdmetrics/single_table/privacy/ens.py  should be deleted

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
        'key1': np.random.choice(['a', 'b', 'c', 'd', 'e'], 20),
        'key2': np.random.randint(0, 5, size=20),
        'sensitive1': np.random.choice(['f', 'g', 'h', 'i', 'j'], 20),
        'sensitive2': np.random.randint(5, 10, size=20)
    })

def cat_good_synthetic_data():
    return pd.DataFrame({
        'key1': np.random.choice(['a', 'b', 'c', 'd', 'e'], 20),
        'key2': np.random.randint(0, 5, size=20),
        'sensitive1': np.random.choice(['a', 'b', 'c', 'd', 'e'], 20),
        'sensitive2': np.random.randint(0, 5, size=20)
    })

def cat_bad_synthetic_data():
    return pd.DataFrame({
        'key1': ['a', 'b', 'c', 'd', 'e'] * 20,
        'key2': [0, 1, 2, 3, 4] * 20,
        'sensitive1': ['a', 'b', 'c', 'e', 'd'] * 20,
        'sensitive2': [0, 1, 2, 3, 4] * 20
    })


@pytest.mark.parametrize('metric', Cat_metrics.values())
def test_categoricals_non_ens(metric):
    if metric !=CategoricalENS:  # ENS is special since it requires additional args to work
        perfect = metric.compute(cat_real_data(), cat_perfect_synthetic_data(),
                key_fields=['key1', 'key2'], sensitive_fields=['sensitive1', 'sensitive2'])

        good = metric.compute(cat_real_data(), cat_good_synthetic_data(),
                key_fields=['key1', 'key2'], sensitive_fields=['sensitive1', 'sensitive2'])

        bad = metric.compute(cat_real_data(), cat_bad_synthetic_data(),
                key_fields=['key1', 'key2'], sensitive_fields=['sensitive1', 'sensitive2'])

        horrible = metric.compute(cat_real_data(), cat_real_data(),
                key_fields=['key1', 'key2'], sensitive_fields=['sensitive1', 'sensitive2'])

        assert metric.min_value <= perfect <= good <= bad <= horrible <= metric.max_value

def test_categorical_ens():
    model_kwargs = {'attackers': [CategoricalNBAttacker, CategoricalRFAttacker, CategoricalKNNAttacker]}
    perfect =CategoricalENS.compute(cat_real_data(), cat_perfect_synthetic_data(),
        key_fields=['key1', 'key2'], sensitive_fields=['sensitive1', 'sensitive2'], model_kwargs=model_kwargs)

    good =CategoricalENS.compute(cat_real_data(), cat_good_synthetic_data(),
        key_fields=['key1', 'key2'], sensitive_fields=['sensitive1', 'sensitive2'], model_kwargs=model_kwargs)

    bad =CategoricalENS.compute(cat_real_data(), cat_bad_synthetic_data(),
        key_fields=['key1', 'key2'], sensitive_fields=['sensitive1', 'sensitive2'], model_kwargs=model_kwargs)

    horrible =CategoricalENS.compute(cat_real_data(), cat_real_data(),
        key_fields=['key1', 'key2'], sensitive_fields=['sensitive1', 'sensitive2'], model_kwargs=model_kwargs)

    assertCategoricalENS.min_value <= perfect <= good <= bad <= horrible <=CategoricalENS.max_value


def numerical_real_data():
    return pd.DataFrame({
        'key1': [0.0, 1.0, 2.0, 3.0, 4.0] * 4,
        'key2': [-0.0, -1.0, -2.0, -3.0, -4.0] * 4,
        'sensitive1': [0.0, 1.0, 2.0, 3.0, 4.0] * 4,
        'sensitive2': [-0.0, -1.0, -2.0, -3.0, -4.0] * 4
    })

def numerical_good_synthetic_data():
    return pd.DataFrame({
        'key1': np.random.normal(size=20),
        'key2': np.random.normal(size=20),
        'sensitive1': np.random.normal(100, size=20),
        'sensitive2': np.random.normal(-100, size=20)
    })

def numerical_bad_synthetic_data():
    return pd.DataFrame({
        'key1': np.random.normal(size=20),
        'key2': np.random.normal(size=20),
        'sensitive1': np.random.normal(size=20),
        'sensitive2': np.random.normal(size=20)
    })


@pytest.mark.parametrize('metric', Num_metrics.values())
def test_num(metric):
    good = metric.compute(numerical_real_data(), numerical_good_synthetic_data(),
        key_fields=['key1', 'key2'], sensitive_fields=['sensitive1', 'sensitive2'])

    bad = metric.compute(numerical_real_data(), numerical_bad_synthetic_data(),
        key_fields=['key1', 'key2'], sensitive_fields=['sensitive1', 'sensitive2'])

    horrible = metric.compute(numerical_real_data(), numerical_real_data(),
        key_fields=['key1', 'key2'], sensitive_fields=['sensitive1', 'sensitive2'])

    print(good)
    print(bad)
    print(horrible)

    assert metric.min_value <= horrible <= bad <= good <= metric.max_value
