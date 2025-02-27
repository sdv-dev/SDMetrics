import numpy as np
import pandas as pd
import pytest

from sdmetrics.single_table import BNLikelihood, BNLogLikelihood


@pytest.fixture
def real_data():
    return pd.DataFrame({
        'a': ['a', 'b', 'a', 'b', 'a', 'b'],
        'b': ['c', 'd', 'c', 'd', 'c', 'd'],
        'c': [True, False, True, False, True, False],
        'd': [1, 2, 3, 4, 5, 6],
        'e': [10, 2, 3, 4, 5, 6],
    })


@pytest.fixture
def synthetic_data():
    return pd.DataFrame({
        'a': ['a', 'b', 'b', 'b', 'a', 'b'],
        'b': ['d', 'd', 'c', 'd', 'c', 'd'],
        'c': [False, False, True, False, True, False],
        'd': [4, 2, 3, 4, 5, 6],
        'e': [12, 2, 3, 4, 5, 6],
    })


@pytest.fixture
def metadata():
    return {
        'columns': {
            'a': {'sdtype': 'categorical'},
            'b': {'sdtype': 'categorical'},
            'c': {'sdtype': 'boolean'},
            'd': {'sdtype': 'categorical'},
            'e': {'sdtype': 'numerical'},
        }
    }


class TestBNLikelihood:
    def test_compute(self, real_data, synthetic_data, metadata):
        """Test the ``compute``method."""
        # Setup
        np.random.seed(42)
        metric = BNLikelihood()

        # Run
        result = metric.compute(real_data, synthetic_data, metadata)

        # Assert
        assert result == 0.111111104


class TestBNLogLikelihood:
    def test_compute(self, real_data, synthetic_data, metadata):
        """Test the ``compute``method."""
        # Setup
        np.random.seed(42)
        metric = BNLogLikelihood()

        # Run
        result = metric.compute(real_data, synthetic_data, metadata)

        # Assert
        assert result == -7.3347335
