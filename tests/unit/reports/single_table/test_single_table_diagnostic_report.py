import pickle
from unittest.mock import Mock, call, mock_open, patch

import pandas as pd

from sdmetrics.reports.single_table import DiagnosticReport


class TestDiagnosticReport:

    def test___init__(self):
        """Test the ``__init__`` method.

        Expect that the correct attributes are initialized.
        """
        # Run
        report = DiagnosticReport()

        # Assert
        assert report._metric_results == {}
        assert report._metric_averages == {}
        assert report._results == {}

    def test_generate(self):
        """Test the ``generate`` method. Expect that the single-table metrics are called."""
        # Setup
        real_data = pd.DataFrame({'col1': [1, 2, 3], 'col2': ['a', 'b', 'c']})
        synthetic_data = pd.DataFrame({'col1': [2, 2, 3], 'col2': ['b', 'a', 'c']})
        metadata = {'fields': {'col1': {'type': 'numerical'}, 'col2': {'type': 'categorical'}}}
        range_coverage = Mock()
        range_coverage.__name__ = 'RangeCoverage'
        range_coverage.compute_breakdown.return_value = {
            'col1': {'score': 0.1},
            'col2': {'score': 0.2},
        }

        category_coverage = Mock()
        category_coverage.__name__ = 'CategoryCoverage'
        category_coverage.compute_breakdown.return_value = {
            'col1': {'score': 0.1},
            'col2': {'score': 0.2},
        }

        new_row_synth = Mock()
        new_row_synth.__name__ = 'NewRowSynthesis'
        new_row_synth.compute_breakdown.return_value = {
            'score': 0.1,
        }

        boundary_adherence = Mock()
        boundary_adherence.__name__ = 'BoundaryAdherence'
        boundary_adherence.compute_breakdown.return_value = {
            'col1': {'score': 0.1},
            'col2': {'score': 0.2},
        }
        metrics_mock = {
            'Coverage': [range_coverage, category_coverage],
            'Synthesis': [new_row_synth],
            'Boundaries': [boundary_adherence],
        }

        # Run
        with patch.object(
            DiagnosticReport,
            'METRICS',
            metrics_mock,
        ):
            report = DiagnosticReport()
            report.generate(real_data, synthetic_data, metadata)

        # Assert
        range_coverage.compute_breakdown.assert_called_once_with(
            real_data, synthetic_data, metadata)
        category_coverage.compute_breakdown.assert_called_once_with(
            real_data, synthetic_data, metadata)
        new_row_synth.compute_breakdown.assert_called_once_with(
            real_data, synthetic_data, metadata)
        boundary_adherence.compute_breakdown.assert_called_once_with(
            real_data, synthetic_data, metadata)
        assert report._metric_averages == {
            'RangeCoverage': 0.15000000000000002,
            'CategoryCoverage': 0.15000000000000002,
            'NewRowSynthesis': 0.1,
            'BoundaryAdherence': 0.15000000000000002,
        }

    def test_generate_with_errored_metric(self):
        """Test the ``generate`` method when the is a metric that has an error.

        Expect that the single-table metrics are called. Expect that the results are computed
        without the error-ed out metric.
        """
        # Setup
        real_data = pd.DataFrame({'col1': [1, 2, 3], 'col2': ['a', 'b', 'c']})
        synthetic_data = pd.DataFrame({'col1': [2, 2, 3], 'col2': ['b', 'a', 'c']})
        range_coverage = Mock()
        range_coverage.__name__ = 'RangeCoverage'
        range_coverage.compute_breakdown.return_value = {
            'col1': {'score': 0.1},
            'col2': {'score': 0.2},
        }

        category_coverage = Mock()
        category_coverage.__name__ = 'CategoryCoverage'
        category_coverage.compute_breakdown.return_value = {
            'col1': {'score': 0.1},
            'col2': {'score': 0.2},
        }

        new_row_synth = Mock()
        new_row_synth.__name__ = 'NewRowSynthesis'
        new_row_synth.compute_breakdown.return_value = {
            'score': 0.1,
        }

        boundary_adherence = Mock()
        boundary_adherence.__name__ = 'BoundaryAdherence'
        boundary_adherence.compute_breakdown.return_value = {
            'col1': {'score': 0.1},
            'col2': {'error': 'test error'},
        }
        metrics_mock = {
            'Coverage': [range_coverage, category_coverage],
            'Synthesis': [new_row_synth],
            'Boundaries': [boundary_adherence],
        }
        metadata = {'fields': {'col1': {'type': 'numerical'}, 'col2': {'type': 'categorical'}}}

        # Run
        with patch.object(
            DiagnosticReport,
            'METRICS',
            metrics_mock,
        ):
            report = DiagnosticReport()
            report.generate(real_data, synthetic_data, metadata)

        # Assert
        range_coverage.compute_breakdown.assert_called_once_with(
            real_data, synthetic_data, metadata)
        category_coverage.compute_breakdown.assert_called_once_with(
            real_data, synthetic_data, metadata)
        new_row_synth.compute_breakdown.assert_called_once_with(
            real_data, synthetic_data, metadata)
        boundary_adherence.compute_breakdown.assert_called_once_with(
            real_data, synthetic_data, metadata)

    def test_get_results(self):
        """Test the ``get_results`` method.

        Expect that the overall diagnostic results are returned.
        """
        # Setup
        report = DiagnosticReport()
        mock_results = {'test_entry': 'test_result'}
        report._results = mock_results

        # Run
        results = report.get_results()

        # Assert
        assert results == mock_results

    def test_get_properties(self):
        """Test the ``get_properties`` method.

        Expect that the property score breakdown is returned.
        """
        # Setup
        report = DiagnosticReport()
        report._metric_averages = {
            'RangeCoverage': 0.1,
            'CategoryCoverage': 0.2,
            'NewRowSynthesis': 0.3,
            'BoundaryAdherence': 0.4,
        }

        # Run
        properties = report.get_properties()

        # Assert
        assert properties == {
            'Synthesis': 0.3,
            'Coverage': 0.15000000000000002,
            'Boundaries': 0.4,
        }

    @patch('sdmetrics.reports.single_table.diagnostic_report.pkg_resources.get_distribution')
    @patch('sdmetrics.reports.single_table.diagnostic_report.pickle')
    def test_save(self, pickle_mock, get_distribution_mock):
        """Test the ``save`` method. Expect that the instance is passed to pickle."""
        # Setup
        report = Mock()
        open_mock = mock_open(read_data=pickle.dumps('test'))

        # Run
        with patch('sdmetrics.reports.single_table.diagnostic_report.open', open_mock):
            DiagnosticReport.save(report, 'test-file.pkl')

        # Assert
        get_distribution_mock.assert_called_once_with('sdmetrics')
        open_mock.assert_called_once_with('test-file.pkl', 'wb')
        pickle_mock.dump.assert_called_once_with(report, open_mock())
        assert report._package_version == get_distribution_mock.return_value.version

    @patch('sdmetrics.reports.single_table.diagnostic_report.pkg_resources.get_distribution')
    @patch('sdmetrics.reports.single_table.diagnostic_report.pickle')
    def test_load(self, pickle_mock, get_distribution_mock):
        """Test the ``load`` method.

        Expect that the report's load method is called with the expected args.
        """
        # Setup
        open_mock = mock_open(read_data=pickle.dumps('test'))
        report = Mock()
        pickle_mock.load.return_value = report
        report._package_version = get_distribution_mock.return_value.version

        # Run
        with patch('sdmetrics.reports.single_table.diagnostic_report.open', open_mock):
            loaded = DiagnosticReport.load('test-file.pkl')

        # Assert
        open_mock.assert_called_once_with('test-file.pkl', 'rb')
        assert loaded == pickle_mock.load.return_value

    @patch('sdmetrics.reports.single_table.diagnostic_report.warnings')
    @patch('sdmetrics.reports.single_table.diagnostic_report.pkg_resources.get_distribution')
    @patch('sdmetrics.reports.single_table.diagnostic_report.pickle')
    def test_load_mismatched_versions(self, pickle_mock, get_distribution_mock, warnings_mock):
        """Test the ``load`` method with mismatched sdmetrics versions.

        Expect that the report's load method is called with the expected args.
        """
        # Setup
        open_mock = mock_open(read_data=pickle.dumps('test'))
        report = Mock()
        pickle_mock.load.return_value = report
        report._package_version = 'previous_version'
        get_distribution_mock.return_value.version = 'new_version'

        # Run
        with patch('sdmetrics.reports.single_table.diagnostic_report.open', open_mock):
            loaded = DiagnosticReport.load('test-file.pkl')

        # Assert
        open_mock.assert_called_once_with('test-file.pkl', 'rb')
        warnings_mock.warn.assert_called_once_with(
            'The report was created using SDMetrics version `previous_version` but you are '
            'currently using version `new_version`. Some features may not work as intended.'
        )
        assert loaded == pickle_mock.load.return_value

    def test_get_details(self):
        """Test the ``get_details`` method.

        Expect that the details of the desired property is returned.
        """
        # Setup
        report = DiagnosticReport()
        report._metric_results = {
            'RangeCoverage': {
                'col1': {'score': 0.1},
                'col2': {'score': 0.2},
            },
            'CategoryCoverage': {
                'col1': {'score': 0.3},
                'col2': {'score': 0.4},
            }
        }

        # Run
        out = report.get_details('Coverage')

        # Assert
        pd.testing.assert_frame_equal(
            out,
            pd.DataFrame({
                'Column': ['col1', 'col2', 'col1', 'col2'],
                'Metric': [
                    'RangeCoverage',
                    'RangeCoverage',
                    'CategoryCoverage',
                    'CategoryCoverage',
                ],
                'Diagnostic Score': [0.1, 0.2, 0.3, 0.4],
            })
        )

    def test_get_details_synthesis(self):
        """Test the ``get_details`` method wit synthesis metrics.

        Expect that the details of the desired property is returned.
        """
        # Setup
        report = DiagnosticReport()
        report._metric_results = {
            'NewRowSynthesis': {'score': 0.1},
        }

        # Run
        out = report.get_details('Synthesis')

        # Assert
        pd.testing.assert_frame_equal(
            out,
            pd.DataFrame({
                'Metric': ['NewRowSynthesis'],
                'Diagnostic Score': [0.1],
            })
        )

    def test__print_result(self):
        """Test the ``_print_results`` method. Expect that the correct messages are written."""
        # Setup
        report = DiagnosticReport()
        report._metric_averages = {'BoundaryAdherence': 0.6}
        mock_out = Mock()

        # Run
        report._print_results(mock_out)

        # Assert
        mock_out.write.assert_has_calls([
            call('WARNING:\n'),
            call('! More than 10% the synthetic data does not follow the min/max boundaries '
                 'set by the real data'),
        ])
