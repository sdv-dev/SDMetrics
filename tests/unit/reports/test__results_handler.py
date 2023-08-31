from unittest.mock import Mock, call, patch

import pandas as pd
import pytest

from sdmetrics.reports._results_handler import (
    BaseResultsHandler, DiagnosticReportResultsHandler, QualityReportResultsHandler)


class TestBaseResultsHandler():
    def test_print_results(self):
        """Test that base print results raises an error."""
        # Setup
        handler = BaseResultsHandler()

        # Run and Assert
        with pytest.raises(NotImplementedError):
            handler.print_results(Mock(), Mock())


class TestDiagnosticReportResultsHandler():
    @patch('sys.stdout.write')
    def test_print_results_verbose_true(self, mock_write):
        """Test that print results prints to std.out when verbose is True."""
        # Setup
        properties = {
            'Coverage': Mock(),
            'Boundary': Mock(),
            'Synthesis': Mock()
        }
        properties['Coverage'].details = pd.DataFrame({
            'Metric': ['CategoryCoverage', 'RangeCoverage', 'CategoryCoverage'],
            'Score': [0.1, 0.2, 0.3]
        })
        properties['Boundary'].details = pd.DataFrame({
            'Metric': ['BoundaryAdherence', 'BoundaryAdherence', 'BoundaryAdherence'],
            'Score': [0.5, 0.6, 0.7]
        })
        properties['Synthesis'].details = pd.DataFrame({
            'Metric': ['NewRowSynthesis'],
            'Score': [1.0]
        })
        handler = DiagnosticReportResultsHandler()

        # Run
        handler.print_results(properties, True)

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

    @patch('sys.stdout.write')
    def test_print_results_verbose_false(self, mock_write):
        """Test that print results just stortes results when verbose is False."""
        # Setup
        properties = {
            'Coverage': Mock(),
            'Boundary': Mock(),
            'Synthesis': Mock()
        }
        properties['Coverage'].details = pd.DataFrame({
            'Metric': ['CategoryCoverage', 'RangeCoverage', 'CategoryCoverage'],
            'Score': [0.1, 0.2, 0.3]
        })
        properties['Boundary'].details = pd.DataFrame({
            'Metric': ['BoundaryAdherence', 'BoundaryAdherence', 'BoundaryAdherence'],
            'Score': [0.5, 0.6, 0.7]
        })
        properties['Synthesis'].details = pd.DataFrame({
            'Metric': ['NewRowSynthesis'],
            'Score': [1.0]
        })
        handler = DiagnosticReportResultsHandler()

        # Run
        handler.print_results(properties, False)

        # Assert
        expected_results = {
            'SUCCESS': ['Over 90% of the synthetic rows are not copies of the real data'],
            'WARNING': [
                'More than 10% the synthetic data does not follow the min/max '
                'boundaries set by the real data'
            ],
            'DANGER': [
                'The synthetic data is missing more than 50% of the categories'
                ' present in the real data',
                'The synthetic data is missing more than 50% of the numerical'
                ' ranges present in the real data'
            ]
        }
        mock_write.assert_not_called()
        assert handler.results == expected_results


class TestQualityReportResultsHandler():
    @patch('sys.stdout.write')
    def test_print_results_verbose_true(self, mock_write):
        """Test the results are printed if verbose is True."""
        # Setup
        score = 0.5
        properties = {
            'Column Shapes': Mock(_compute_average=Mock(return_value=0.6)),
            'Column Pair Trends': Mock(_compute_average=Mock(return_value=0.4))
        }
        handler = QualityReportResultsHandler()
        # Run
        handler.print_results(properties, score, True)

        # Assert
        calls = [
            call('\nOverall Quality Score: 50.0%\n\n'),
            call('Properties:\n'),
            call('- Column Shapes: 60.0%\n'),
            call('- Column Pair Trends: 40.0%\n'),
        ]
        mock_write.assert_has_calls(calls, any_order=True)

    @patch('sys.stdout.write')
    def test_print_results_verbose_false(self, mock_write):
        """Test the results are not printed if verbose is False."""
        # Setup
        score = 0.5
        properties = {
            'Column Shapes': Mock(_compute_average=Mock(return_value=0.6)),
            'Column Pair Trends': Mock(_compute_average=Mock(return_value=0.4))
        }
        handler = QualityReportResultsHandler()
        # Run
        handler.print_results(properties, score, False)

        # Assert
        mock_write.assert_not_called()
