import numpy as np
import pandas as pd
import pytest

from sdmetrics.single_column.statistical.kstest import KSTest


@pytest.mark.parametrize("array_like", [np.array, pd.Series])
def test_max(array_like):
    data = array_like(np.random.normal(size=1000))
    output = KSTest.compute(data, data)
    normalized = KSTest.normalize(output)

    assert output == 1
    assert normalized == 1


@pytest.mark.parametrize("array_like", [np.array, pd.Series])
def test_min(array_like):
    real = array_like(np.random.normal(size=1000))
    synth = array_like(np.random.normal(loc=1000, scale=10, size=1000))
    output = KSTest.compute(real, synth)
    normalized = KSTest.normalize(output)

    assert output == 0
    assert normalized == 0


@pytest.mark.parametrize("array_like", [np.array, pd.Series])
def test_good(array_like):
    real = array_like(np.random.normal(size=1000))
    synth = array_like(np.random.normal(loc=0.1, size=1000))
    output = KSTest.compute(real, synth)
    normalized = KSTest.normalize(output)

    assert 0.5 < output <= 1.0
    assert 0.5 < normalized <= 1.0


@pytest.mark.parametrize("array_like", [np.array, pd.Series])
def test_bad(array_like):
    real = array_like(np.random.normal(size=1000))
    synth = array_like(np.random.normal(loc=3, scale=3, size=1000))
    output = KSTest.compute(real, synth)
    normalized = KSTest.normalize(output)

    assert 0.0 <= output < 0.5
    assert 0.0 <= normalized < 0.5
