import numpy as np
import pandas as pd
import pytest

from sdmetrics.single_column.statistical.kscomplement import KSComplement


@pytest.mark.parametrize('array_like', [np.array, pd.Series])
def test_max(array_like):
    data = array_like(np.random.normal(size=1000))
    output = KSComplement.compute(data, data)
    normalized = KSComplement.normalize(output)

    assert output == 1
    assert normalized == 1


@pytest.mark.parametrize('array_like', [np.array, pd.Series])
def test_min(array_like):
    real = array_like(np.random.normal(size=1000))
    synth = array_like(np.random.normal(loc=1000, scale=10, size=1000))
    output = KSComplement.compute(real, synth)
    normalized = KSComplement.normalize(output)

    assert output == 0
    assert normalized == 0


@pytest.mark.parametrize('array_like', [np.array, pd.Series])
def test_good(array_like):
    real = array_like(np.random.normal(size=1000))
    synth = array_like(np.random.normal(loc=0.1, size=1000))
    output = KSComplement.compute(real, synth)
    normalized = KSComplement.normalize(output)

    assert 0.5 < output <= 1.0
    assert 0.5 < normalized <= 1.0


@pytest.mark.parametrize('array_like', [np.array, pd.Series])
def test_bad(array_like):
    real = array_like(np.random.normal(size=1000))
    synth = array_like(np.random.normal(loc=3, scale=3, size=1000))
    output = KSComplement.compute(real, synth)
    normalized = KSComplement.normalize(output)

    assert 0.0 <= output < 0.5
    assert 0.0 <= normalized < 0.5


def test_one_float_value():
    """Test KSComplement.compute when both data have the same float values GH#652."""
    # Setup
    real = pd.Series([0.3 - 0.2])
    synth = pd.Series([0.2 - 0.1])

    # Run
    output = KSComplement.compute(real, synth)

    # Assert
    assert output == 1
