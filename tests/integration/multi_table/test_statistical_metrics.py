import random

import numpy as np
import pandas as pd
import pytest

from sdmetrics.multi_table.statistical import (
    CardinalityShapeSimilarity, CardinalityStatisticSimilarity)

METRICS = [CardinalityShapeSimilarity, CardinalityStatisticSimilarity]


def real_data():
    parent = pd.DataFrame({
        'id': range(60),
        'a': np.random.normal(size=60),
        'b': np.random.randint(0, 10, size=60),
        'c': ['a', 'b', 'b', 'c', 'c', 'c'] * 10,
        'd': [True, True, True, True, True, False] * 10,
    })
    child = pd.DataFrame({
        'parent_id': [random.randint(0, 60) for _ in range(600)],
        'a': np.random.normal(size=600),
        'b': np.random.randint(0, 10, size=600),
        'c': ['a', 'b', 'b', 'c', 'c', 'c'] * 100,
        'd': [True, True, True, True, True, False] * 100,
    })
    return {'parent': parent, 'child': child}


def good_data():
    parent = pd.DataFrame({
        'id': range(60),
        'a': np.random.normal(loc=0.01, size=60),
        'b': np.random.randint(1, 11, size=60),
        'c': ['a', 'b', 'b', 'c', 'c', 'c'] * 10,
        'd': [True, True, True, True, True, False] * 10,
    })
    child = pd.DataFrame({
        'parent_id': [random.randint(0, 60) for _ in range(600)],
        'a': np.random.normal(loc=0.01, size=600),
        'b': np.random.randint(1, 11, size=600),
        'c': ['a', 'b', 'b', 'c', 'c', 'c'] * 100,
        'd': [True, True, True, True, True, False] * 100,
    })
    return {'parent': parent, 'child': child}


def bad_data():
    parent = pd.DataFrame({
        'id': range(60),
        'a': np.random.normal(loc=5, scale=3, size=60),
        'b': np.random.randint(5, 15, size=60),
        'c': ['a', 'a', 'a', 'a', 'b', 'b'] * 10,
        'd': [True, False, False, False, False, False] * 10,
    })
    child = pd.DataFrame({
        'parent_id': [random.randint(0, 60) for _ in range(600)],
        'a': np.random.normal(loc=5, scale=3, size=600),
        'b': np.random.randint(5, 15, size=600),
        'c': ['a', 'a', 'a', 'a', 'b', 'b'] * 100,
        'd': [True, False, False, False, False, False] * 100,
    })
    return {'parent': parent, 'child': child}


METADATA = {
    'tables': {
        'parent': {
            'columns': {
                'id': {}
            }
        },
        'child': {
            'columns': {
                'parent_id': {},
            }
        },
    },
    'relationships': [
        {
            'parent_table_name': 'parent',
            'parent_primary_key': 'id',
            'child_table_name': 'child',
            'child_foreign_key': 'parent_id'
        }
    ]
}


@pytest.mark.parametrize('metric', METRICS)
def test_good(metric):
    output = metric.compute(real_data(), good_data(), metadata=METADATA)
    normalized = metric.normalize(output)

    assert 0 <= output <= 1
    assert 0 <= normalized <= 1


@pytest.mark.parametrize('metric', METRICS)
def test_bad(metric):
    output = metric.compute(real_data(), bad_data(), metadata=METADATA)
    normalized = metric.normalize(output)

    assert 0 <= output <= 1
    assert 0 <= normalized <= 1
