import numpy as np
import pandas as pd
import pytest

from sdmetrics.single_column.statistical.cstest import CSTest


@pytest.mark.parametrize("array_like", [np.array, pd.Series])
def test_max(array_like):
    data = array_like(['a', 'b', 'c', 'd'])
    output = CSTest.compute(data, data)

    assert output == 1


@pytest.mark.parametrize("array_like", [np.array, pd.Series])
def test_min(array_like):
    real = array_like([True, True, True, True])
    synth = array_like([False, False, False, False])
    output = CSTest.compute(real, synth)

    assert output == 0


@pytest.mark.parametrize("array_like", [np.array, pd.Series])
def test_middle(array_like):
    real = array_like(['a', 'a', 'b', 'c'])
    synth = array_like(['a', 'b', 'b', 'b'])
    output = CSTest.compute(real, synth)

    assert 0.0 < output < 1.0
