import numpy as np
import pandas as pd
import pytest

from sdmetrics.single_column.statistical.kstest import KSTest


@pytest.mark.parametrize("array_like", [np.array, pd.Series])
def test_max(array_like):
    data = array_like([0] * 1000)
    output = KSTest.compute(data, data)

    assert output == 1


@pytest.mark.parametrize("array_like", [np.array, pd.Series])
def test_min(array_like):
    real = array_like([0] * 1000)
    synth = array_like([1] * 1000)
    output = KSTest.compute(real, synth)

    assert output == 0


@pytest.mark.parametrize("array_like", [np.array, pd.Series])
def test_middle(array_like):
    real = array_like([1, 2, 2, 1])
    synth = array_like([0, 1, 1, 0])
    output = KSTest.compute(real, synth)

    assert 0.0 < output < 1.0
