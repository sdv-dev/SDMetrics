from unittest.mock import call, patch

import pandas as pd
import pytest

from sdmetrics.reports.single_table import DiagnosticReport
from sdmetrics.reports.single_table._properties import Boundary, Coverage, Synthesis


class TestDiagnosticReport:

    def test___init__(self):
        """Test the ``__init__`` method."""
        # Run
        report = DiagnosticReport()

        # Assert
        assert report._overall_score is None
        assert report.is_generated is False
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

    def test_get_results(self):
        """Test the ``get_results`` method."""
        # Setup
        diagnostic_report = DiagnosticReport()
        diagnostic_report.results = {'SUCCESS': ['Test']}
        diagnostic_report.is_generated = True

        # Run
        results = diagnostic_report.get_results()

        # Assert
        assert results == {'SUCCESS': ['Test']}

    @patch('sys.stdout.write')
    def test__print_results(self, mock_write):
        """Test the ``_print_results`` method."""
        # Setup
        diagnostic_report = DiagnosticReport()
        diagnostic_report = DiagnosticReport()
        diagnostic_report._properties['Coverage']._details = pd.DataFrame({
            'Metric': ['CategoryCoverage', 'RangeCoverage', 'CategoryCoverage'],
            'Score': [0.1, 0.2, 0.3]
        })
        diagnostic_report._properties['Boundary']._details = pd.DataFrame({
            'Metric': ['BoundaryAdherence', 'BoundaryAdherence', 'BoundaryAdherence'],
            'Score': [0.5, 0.6, 0.7]
        })
        diagnostic_report._properties['Synthesis']._details = pd.DataFrame({
            'Metric': ['NewRowSynthesis'],
            'Score': [1.0]
        })

        # Run
        diagnostic_report._print_results()

        # Assert
        calls = [
            call('\nDiagnostic Results:\n'),
            call('\nSUCCESS:\n'),
            call('✓ Over 90% of the synthetic rows are not copies of the real data\n'),
            call('\nWARNING:\n'),
            call(
                '! More than 10% the synthetic data does not follow the min/max '
                'boundaries set by the real data\n'
            ),
            call('\nDANGER:\n'),
            call(
                'x The synthetic data is missing more than 50% of the categories'
                ' present in the real data\n'
            ),
            call(
                'x The synthetic data is missing more than 50% of the numerical'
                ' ranges present in the real data\n'
            )
        ]

        mock_write.assert_has_calls(calls, any_order=True)
