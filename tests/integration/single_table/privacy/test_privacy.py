import numpy as np
import pandas as pd
import pytest

from sdmetrics.single_table.privacy import (
    CategoricalEnsemble, CategoricalPrivacyMetric, NumericalPrivacyMetric)
from sdmetrics.single_table.privacy.categorical_sklearn import (
    CategoricalKNNAttacker, CategoricalNBAttacker, CategoricalRFAttacker)

categorical_metrics = CategoricalPrivacyMetric.get_subclasses()
numerical_metrics = NumericalPrivacyMetric.get_subclasses()


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


@pytest.mark.parametrize('metric', categorical_metrics.values())
def test_categoricals_non_ens(metric):
    if metric != CategoricalEnsemble:  # Ensemble needs additional args to work
        perfect = metric.compute(
            cat_real_data(), cat_perfect_synthetic_data(),
            key_fields=['key1', 'key2'], sensitive_fields=['sensitive1', 'sensitive2']
        )

        good = metric.compute(
            cat_real_data(), cat_good_synthetic_data(),
            key_fields=['key1', 'key2'], sensitive_fields=['sensitive1', 'sensitive2']
        )

        bad = metric.compute(
            cat_real_data(), cat_bad_synthetic_data(),
            key_fields=['key1', 'key2'], sensitive_fields=['sensitive1', 'sensitive2']
        )

        horrible = metric.compute(
            cat_real_data(), cat_real_data(),
            key_fields=['key1', 'key2'], sensitive_fields=['sensitive1', 'sensitive2']
        )

        assert metric.min_value <= horrible <= bad <= good <= perfect <= metric.max_value


def test_categorical_ens():
    model_kwargs = {
        'attackers': [CategoricalNBAttacker, CategoricalRFAttacker, CategoricalKNNAttacker]
    }
    perfect = CategoricalEnsemble.compute(
        cat_real_data(), cat_perfect_synthetic_data(),
        key_fields=['key1', 'key2'], sensitive_fields=['sensitive1', 'sensitive2'],
        model_kwargs=model_kwargs
    )

    good = CategoricalEnsemble.compute(
        cat_real_data(), cat_good_synthetic_data(),
        key_fields=['key1', 'key2'], sensitive_fields=['sensitive1', 'sensitive2'],
        model_kwargs=model_kwargs
    )

    bad = CategoricalEnsemble.compute(
        cat_real_data(), cat_bad_synthetic_data(),
        key_fields=['key1', 'key2'], sensitive_fields=['sensitive1', 'sensitive2'],
        model_kwargs=model_kwargs
    )

    horrible = CategoricalEnsemble.compute(
        cat_real_data(), cat_real_data(),
        key_fields=['key1', 'key2'], sensitive_fields=['sensitive1', 'sensitive2'],
        model_kwargs=model_kwargs
    )

    assert (CategoricalEnsemble.min_value <= horrible <= bad
            <= good <= perfect <= CategoricalEnsemble.max_value)


def numerical_real_data():
    return pd.DataFrame({
        'key1': [0.0, 0.1, 0.2, 0.3, 0.4] * 4,
        'key2': [-0.0, -0.1, -0.2, -0.3, -0.4] * 4,
        'sensitive1': [0.0, 0.1, 0.2, 0.3, 0.4] * 4,
        'sensitive2': [-0.0, -0.1, -0.2, -0.3, -0.4] * 4
    })


def numerical_good_synthetic_data():
    return pd.DataFrame({
        'key1': np.random.normal(loc=0.2, scale=0.1, size=20),
        'key2': np.random.normal(loc=-0.2, scale=0.1, size=20),
        'sensitive1': np.random.normal(loc=10.0, size=20),
        'sensitive2': np.random.normal(loc=-10.0, size=20)
    })


def numerical_bad_synthetic_data():
    return pd.DataFrame({
        'key1': np.random.normal(loc=0.2, scale=0.1, size=20),
        'key2': np.random.normal(loc=-0.2, scale=0.1, size=20),
        'sensitive1': np.random.normal(size=20),
        'sensitive2': np.random.normal(size=20)
    })


@pytest.mark.parametrize('metric', numerical_metrics.values())
def test_num(metric):
    good = metric.compute(
        numerical_real_data(), numerical_good_synthetic_data(),
        key_fields=['key1', 'key2'], sensitive_fields=['sensitive1', 'sensitive2']
    )

    bad = metric.compute(
        numerical_real_data(), numerical_bad_synthetic_data(),
        key_fields=['key1', 'key2'], sensitive_fields=['sensitive1', 'sensitive2']
    )

    horrible = metric.compute(
        numerical_real_data(), numerical_real_data(),
        key_fields=['key1', 'key2'], sensitive_fields=['sensitive1', 'sensitive2']
    )

    assert metric.min_value <= horrible <= bad <= good <= metric.max_value


@pytest.mark.parametrize('metric', categorical_metrics.values())
def test_categorical_empty_keys(metric):
    if metric != CategoricalEnsemble:
        with pytest.raises(TypeError):
            metric.compute(cat_real_data(), cat_real_data(), sensitive_fields=['sensitive1'])


@pytest.mark.parametrize('metric', categorical_metrics.values())
def test_categorical_empty_sensitive(metric):
    if metric != CategoricalEnsemble:
        with pytest.raises(TypeError):
            metric.compute(cat_real_data(), cat_real_data(), key_fields=['key1'])


@pytest.mark.parametrize('metric', categorical_metrics.values())
def test_categorical_empty_keys_sensitive(metric):
    if metric != CategoricalEnsemble:
        with pytest.raises(TypeError):
            metric.compute(cat_real_data(), cat_real_data())
