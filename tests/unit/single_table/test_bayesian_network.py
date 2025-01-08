import sys
from unittest.mock import Mock

import pytest

from sdmetrics.single_table import BNLikelihood, BNLogLikelihood


@pytest.fixture
def bad_pomegranate():
    old_pomegranate = getattr(sys.modules, 'pomegranate', None)
    sys.modules['pomegranate'] = pytest
    yield
    if old_pomegranate is not None:
        sys.modules['pomegranate'] = old_pomegranate
    else:
        del sys.modules['pomegranate']


class TestBNLikelihood:
    def test_compute(self, bad_pomegranate):
        """Test that an ``ImportError`` is raised."""
        # Setup
        metric = BNLikelihood()

        # Act and Assert
        expected_message = r'Please install pomegranate with `pip install sdmetrics\[pomegranate\]`'
        with pytest.raises(ImportError, match=expected_message):
            metric.compute(Mock(), Mock())


class TestBNLogLikelihood:
    def test_compute(self, bad_pomegranate):
        """Test that an ``ImportError`` is raised."""
        # Setup
        metric = BNLogLikelihood()

        # Act and Assert
        expected_message = r'Please install pomegranate with `pip install sdmetrics\[pomegranate\]`'
        with pytest.raises(ImportError, match=expected_message):
            metric.compute(Mock(), Mock())
