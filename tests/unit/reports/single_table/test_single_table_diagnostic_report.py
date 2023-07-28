import contextlib
import io
import pickle
from unittest.mock import Mock, call, mock_open, patch

import numpy as np
import pandas as pd
import pytest
import re

from sdmetrics.errors import IncomputableMetricError
from sdmetrics.reports.single_table import DiagnosticReport
from sdmetrics.reports.single_table._properties import (
    Coverage, Boundary, Synthesis
)


class TestDiagnosticReport:

    def test___init__(self):
        """Test the ``__init__`` method."""
        # Run
        report = DiagnosticReport()

        # Assert
        assert report._overall_quality_score is None
        assert report.is_generated == False
        assert isinstance(report._properties['Coverage'], Coverage)
        assert isinstance(report._properties['Boundary'], Boundary)
        assert isinstance(report._properties['Synthesis'], Synthesis)

    def test__get_num_iterations(self):
        """Test the ``_get_num_iterations`` method."""
        # Setup
        report = DiagnosticReport()
        metadata = {'columns': {'a': {}, 'b': {}, 'c': {}}}

        # Run
        num_iterations_coverage = report._get_num_iterations('Coverage', metadata)
        num_iterations_boundaries = report._get_num_iterations('Boundary', metadata)
        num_iterations_synthesis = report._get_num_iterations('Synthesis', metadata)

        expected_error_message = (
            "Invalid property name 'Invalid_property'."
            " Valid property names are 'Coverage', 'Boundary', 'Synthesis'."
        )
        with pytest.raises(ValueError, match=expected_error_message):
            report._get_num_iterations('Invalid_property', metadata)

        # Assert
        assert num_iterations_coverage == 3
        assert num_iterations_boundaries == 3
        assert num_iterations_synthesis == 1