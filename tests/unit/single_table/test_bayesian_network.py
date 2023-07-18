from unittest.mock import Mock

import pytest

from sdmetrics.single_table import BNLikelihood, BNLogLikelihood


class TestBNLikelihood:

    def test_compute(self):
        """Test that an ``ImportError`` is raised."""
        # Setup
        metric = BNLikelihood()

        # Act and Assert
        expected_message = (
            'Please install pomegranate with `pip install pomegranate` on a version of python '
            '< 3.11. This metric is not supported on python versions >= 3.11.'
        )
        with pytest.raises(ImportError, match=expected_message):
            metric.compute(Mock(), Mock())


class TestBNLogLikelihood:

    def test_compute(self):
        """Test that an ``ImportError`` is raised."""
        # Setup
        metric = BNLogLikelihood()

        # Act and Assert
        expected_message = (
            'Please install pomegranate with `pip install pomegranate` on a version of python '
            '< 3.11. This metric is not supported on python versions >= 3.11.'
        )
        with pytest.raises(ImportError, match=expected_message):
            metric.compute(Mock(), Mock())
