import numpy as np
import pandas as pd
import pytest

from sdmetrics.single_column.statistical.cstest import CSTest


@pytest.mark.parametrize("array_like", [np.array, pd.Series])
def test_max(array_like):
    data = array_like(['a', 'b', 'b', 'c', 'c', 'c'] * 100)
    output = CSTest.compute(data, data)

    assert output == 1


@pytest.mark.parametrize("array_like", [np.array, pd.Series])
def test_min(array_like):
    real = array_like(['a', 'b', 'b', 'c', 'c', 'c'] * 100)
    synth = array_like(['d', 'e', 'e', 'f', 'f', 'f'] * 100)
    output = CSTest.compute(real, synth)

    assert output == 0


@pytest.mark.parametrize("array_like", [np.array, pd.Series])
def test_good(array_like):
    real = array_like(['a', 'b', 'b', 'c', 'c', 'c'] * 100)
    synth = array_like(['a', 'b', 'b', 'b', 'c', 'c'] * 100)
    output = CSTest.compute(real, synth)

    assert 0.5 < output <= 1.0


@pytest.mark.parametrize("array_like", [np.array, pd.Series])
def test_bad(array_like):
    real = array_like(['a', 'b', 'b', 'c', 'c', 'c'] * 100)
    synth = array_like(['a', 'a', 'a', 'a', 'b', 'c'] * 100)
    output = CSTest.compute(real, synth)

    assert 0.0 <= output < 0.5
