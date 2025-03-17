import re
from unittest.mock import Mock, patch

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
    @patch.dict('sys.modules', {'pomegranate.bayesian_network': None})
    def test_compute_error(self):
        """Test that an `ImportError` is raised."""
        # Setup
        metric = BNLikelihood()

        # Run and Assert
        expected_message = re.escape(
            'Please install pomegranate with `pip install sdmetrics[pomegranate]`.'
        )
        with pytest.raises(ImportError, match=expected_message):
            metric.compute(Mock(), Mock())

    def test_compute(self, real_data, synthetic_data, metadata):
        """Test the ``compute``method."""
        # Setup
        np.random.seed(42)
        metric = BNLikelihood()

        # Run
        result = metric.compute(real_data, synthetic_data, metadata)

        # Assert
        assert np.isclose(result, 0.1111111044883728, atol=1e-5)


class TestBNLogLikelihood:
    @patch.dict('sys.modules', {'pomegranate.bayesian_network': None})
    def test_compute_error(self):
        """Test that an `ImportError` is raised."""
        # Setup
        metric = BNLogLikelihood()

        # Run and Assert
        expected_message = re.escape(
            'Please install pomegranate with `pip install sdmetrics[pomegranate]`.'
        )
        with pytest.raises(ImportError, match=expected_message):
            metric.compute(Mock(), Mock())

    def test_compute(self, real_data, synthetic_data, metadata):
        """Test the ``compute``method."""
        # Setup
        np.random.seed(42)
        metric = BNLogLikelihood()

        # Run
        result = metric.compute(real_data, synthetic_data, metadata)

        # Assert
        assert np.isclose(result, -7.334733486175537, atol=1e-5)
