import contextlib
import io
import pickle
from unittest.mock import Mock, call, mock_open, patch

import numpy as np
import pandas as pd

from sdmetrics.errors import IncomputableMetricError
from sdmetrics.reports.single_table import QualityReport


class TestQualityReport:

    def test___init__(self):
        """Test the ``__init__`` method.

        Expect that the correct attributes are initialized.
        """
        # Run
        report = QualityReport()

        # Assert
        assert report._overall_quality_score is None
        assert report._metric_results == {}
        assert report._property_breakdown == {}

    @patch('sdmetrics.reports.single_table.quality_report.discretize_and_apply_metric')
    def test_generate(self, mock_discretize_and_apply_metric):
        """Test the ``generate`` method.

        Expect that the single-table metrics are called.

        Setup:
        - Mock the expected single-table metric compute breakdown calls.

        Input:
        - Real data.
        - Synthetic data.
        - Metadata.

        Side Effects:
        - Expect that each single table metric's ``compute_breakdown`` methods are called once.
        - Expect that the ``_overall_quality_score`` and ``_property_breakdown`` attributes
          are populated.
        """
        # Setup
        real_data = pd.DataFrame({'col1': [1, 2, 3], 'col2': ['a', 'b', 'c']})
        synthetic_data = pd.DataFrame({'col1': [2, 2, 3], 'col2': ['b', 'a', 'c']})
        ks_complement_mock = Mock()
        metadata = {
            'columns': {
                'col1': {'sdtype': 'numerical'},
                'col2': {'sdtype': 'categorical'}
            }
        }
        ks_complement_mock.__name__ = 'KSComplement'
        ks_complement_mock.compute_breakdown.return_value = {
            'col1': {'score': 0.1},
            'col2': {'score': 0.2},
        }

        tv_complement_mock = Mock()
        tv_complement_mock.__name__ = 'TVComplement'
        tv_complement_mock.compute_breakdown.return_value = {
            'col1': {'score': 0.1},
            'col2': {'score': 0.2},
        }

        corr_sim_mock = Mock()
        corr_sim_mock.__name__ = 'CorrelationSimilarity'
        corr_sim_mock.compute_breakdown.return_value = {
            ('col1', 'col2'): {'score': 0.1},
            ('col2', 'col3'): {'score': 0.2},
        }

        cont_sim_mock = Mock()
        cont_sim_mock.__name__ = 'ContingencySimilarity'
        cont_sim_mock.compute_breakdown.return_value = {
            ('col1', 'col2'): {'score': 0.1},
            ('col2', 'col3'): {'score': 0.2},
        }
        metrics_mock = {
            'Column Shapes': [ks_complement_mock, tv_complement_mock],
            'Column Pair Trends': [corr_sim_mock, cont_sim_mock],
        }
        mock_discretize_and_apply_metric.return_value = {}

        # Run
        with patch.object(
            QualityReport,
            'METRICS',
            metrics_mock,
        ):
            report = QualityReport()
            report.generate(real_data, synthetic_data, metadata)

        # Assert
        ks_complement_mock.compute_breakdown.assert_called_once_with(
            real_data, synthetic_data, metadata)
        tv_complement_mock.compute_breakdown.assert_called_once_with(
            real_data, synthetic_data, metadata)
        corr_sim_mock.compute_breakdown.assert_called_once_with(
            real_data, synthetic_data, metadata)
        cont_sim_mock.compute_breakdown.assert_called_once_with(
            real_data, synthetic_data, metadata)
        assert report._overall_quality_score == 0.15000000000000002
        assert report._property_breakdown == {
            'Column Shapes': 0.15000000000000002,
            'Column Pair Trends': 0.15000000000000002,
        }

    @patch('sdmetrics.reports.single_table.quality_report.discretize_and_apply_metric')
    def test_generate_verbose_false(self, mock_discretize_and_apply_metric):
        """Test the ``generate`` method with silent mode. Expect that nothing is printed.
        """
        # Setup
        real_data = pd.DataFrame({'col1': [1, 2, 3], 'col2': ['a', 'b', 'c']})
        synthetic_data = pd.DataFrame({'col1': [2, 2, 3], 'col2': ['b', 'a', 'c']})
        ks_complement_mock = Mock()
        metadata = {
            'columns': {
                'col1': {'sdtype': 'numerical'},
                'col2': {'sdtype': 'categorical'}
            }
        }
        ks_complement_mock.__name__ = 'KSComplement'
        ks_complement_mock.compute_breakdown.return_value = {
            'col1': {'score': 0.1},
            'col2': {'score': 0.2},
        }

        tv_complement_mock = Mock()
        tv_complement_mock.__name__ = 'TVComplement'
        tv_complement_mock.compute_breakdown.return_value = {
            'col1': {'score': 0.1},
            'col2': {'score': 0.2},
        }

        corr_sim_mock = Mock()
        corr_sim_mock.__name__ = 'CorrelationSimilarity'
        corr_sim_mock.compute_breakdown.return_value = {
            ('col1', 'col2'): {'score': 0.1},
            ('col2', 'col3'): {'score': 0.2},
        }

        cont_sim_mock = Mock()
        cont_sim_mock.__name__ = 'ContingencySimilarity'
        cont_sim_mock.compute_breakdown.return_value = {
            ('col1', 'col2'): {'score': 0.1},
            ('col2', 'col3'): {'score': 0.2},
        }
        metrics_mock = {
            'Column Shapes': [ks_complement_mock, tv_complement_mock],
            'Column Pair Trends': [corr_sim_mock, cont_sim_mock],
        }
        mock_discretize_and_apply_metric.return_value = {}

        # Run
        prints = io.StringIO()
        with contextlib.redirect_stderr(prints), patch.object(
            QualityReport,
            'METRICS',
            metrics_mock,
        ):
            report = QualityReport()
            report.generate(real_data, synthetic_data, metadata, verbose=False)

        # Assert
        ks_complement_mock.compute_breakdown.assert_called_once_with(
            real_data, synthetic_data, metadata)
        tv_complement_mock.compute_breakdown.assert_called_once_with(
            real_data, synthetic_data, metadata)
        corr_sim_mock.compute_breakdown.assert_called_once_with(
            real_data, synthetic_data, metadata)
        cont_sim_mock.compute_breakdown.assert_called_once_with(
            real_data, synthetic_data, metadata)
        assert report._overall_quality_score == 0.15000000000000002
        assert report._property_breakdown == {
            'Column Shapes': 0.15000000000000002,
            'Column Pair Trends': 0.15000000000000002,
        }
        assert prints.getvalue() == ''

    @patch('sdmetrics.reports.single_table.quality_report.discretize_and_apply_metric')
    def test_generate_empty_column_pairs_results(self, mock_discretize_and_apply_metric):
        """Test the ``generate`` method when there are no column pair results.

        Expect that the single-table metrics are called. Expect that the column pair
        results is NaN but the overall score is not.

        Setup:
        - Mock the expected single-table metric compute breakdown calls.

        Input:
        - Real data.
        - Synthetic data.
        - Metadata.

        Side Effects:
        - Expect that each single table metric's ``compute_breakdown`` methods are called once.
        - Expect that the ``_overall_quality_score`` and ``_property_breakdown`` attributes
          are populated.
        """
        # Setup
        real_data = pd.DataFrame({'col1': [1, 2, 3], 'col2': ['a', 'b', 'c']})
        synthetic_data = pd.DataFrame({'col1': [2, 2, 3], 'col2': ['b', 'a', 'c']})
        ks_complement_mock = Mock()
        metadata = {
            'columns': {
                'col1': {'sdtype': 'numerical'},
                'col2': {'sdtype': 'categorical'}
            }
        }
        ks_complement_mock.__name__ = 'KSComplement'
        ks_complement_mock.compute_breakdown.return_value = {
            'col1': {'score': 0.1},
            'col2': {'score': 0.2},
        }

        tv_complement_mock = Mock()
        tv_complement_mock.__name__ = 'TVComplement'
        tv_complement_mock.compute_breakdown.return_value = {
            'col1': {'score': 0.1},
            'col2': {'score': 0.2},
        }

        corr_sim_mock = Mock()
        corr_sim_mock.__name__ = 'CorrelationSimilarity'
        corr_sim_mock.compute_breakdown.return_value = {}

        cont_sim_mock = Mock()
        cont_sim_mock.__name__ = 'ContingencySimilarity'
        cont_sim_mock.compute_breakdown.return_value = {}

        metrics_mock = {
            'Column Shapes': [ks_complement_mock, tv_complement_mock],
            'Column Pair Trends': [corr_sim_mock, cont_sim_mock],
        }
        mock_discretize_and_apply_metric.return_value = {}

        # Run
        with patch.object(
            QualityReport,
            'METRICS',
            metrics_mock,
        ):
            report = QualityReport()
            report.generate(real_data, synthetic_data, metadata)

        # Assert
        ks_complement_mock.compute_breakdown.assert_called_once_with(
            real_data, synthetic_data, metadata)
        tv_complement_mock.compute_breakdown.assert_called_once_with(
            real_data, synthetic_data, metadata)
        corr_sim_mock.compute_breakdown.assert_called_once_with(
            real_data, synthetic_data, metadata)
        cont_sim_mock.compute_breakdown.assert_called_once_with(
            real_data, synthetic_data, metadata)
        assert report._overall_quality_score == 0.15000000000000002
        assert np.isnan(report._property_breakdown['Column Pair Trends'])

    @patch('sdmetrics.reports.single_table.quality_report.discretize_and_apply_metric')
    def test_generate_with_errored_metric(self, mock_discretize_and_apply_metric):
        """Test the ``generate`` method when the is a metric that has an error.

        Expect that the single-table metrics are called. Expect that the overall score is
        computed without the error-ed out metric.

        Setup:
        - Mock the expected single-table metric compute breakdown calls.

        Input:
        - Real data.
        - Synthetic data.
        - Metadata.

        Side Effects:
        - Expect that each single table metric's ``compute_breakdown`` methods are called once.
        - Expect that the ``_overall_quality_score`` and ``_property_breakdown`` attributes
          are populated.
        """
        # Setup
        real_data = pd.DataFrame({'col1': [1, 2, 3], 'col2': ['a', 'b', 'c']})
        synthetic_data = pd.DataFrame({'col1': [2, 2, 3], 'col2': ['b', 'a', 'c']})
        ks_complement_mock = Mock()
        metadata = {
            'columns': {
                'col1': {'sdtype': 'numerical'},
                'col2': {'sdtype': 'categorical'}
            }
        }
        ks_complement_mock.__name__ = 'KSComplement'
        ks_complement_mock.compute_breakdown.return_value = {
            'col1': {'score': 0.1},
            'col2': {'score': 0.2},
        }

        tv_complement_mock = Mock()
        tv_complement_mock.__name__ = 'TVComplement'
        tv_complement_mock.compute_breakdown.return_value = {
            'col1': {'score': 0.1},
            'col2': {'score': 0.2},
        }

        corr_sim_mock = Mock()
        corr_sim_mock.__name__ = 'CorrelationSimilarity'
        corr_sim_mock.compute_breakdown.return_value = {
            ('col1', 'col2'): {'score': 0.1},
            ('col2', 'col3'): {'score': 0.2},
        }

        cont_sim_mock = Mock()
        cont_sim_mock.__name__ = 'ContingencySimilarity'
        cont_sim_mock.compute_breakdown.return_value = {
            ('col1', 'col2'): {'score': 0.1},
            ('col2', 'col3'): {'error': 'test error'},
        }

        metrics_mock = {
            'Column Shapes': [ks_complement_mock, tv_complement_mock],
            'Column Pair Trends': [corr_sim_mock, cont_sim_mock],
        }
        mock_discretize_and_apply_metric.return_value = {}

        # Run
        with patch.object(
            QualityReport,
            'METRICS',
            metrics_mock,
        ):
            report = QualityReport()
            report.generate(real_data, synthetic_data, metadata)

        # Assert
        ks_complement_mock.compute_breakdown.assert_called_once_with(
            real_data, synthetic_data, metadata)
        tv_complement_mock.compute_breakdown.assert_called_once_with(
            real_data, synthetic_data, metadata)
        corr_sim_mock.compute_breakdown.assert_called_once_with(
            real_data, synthetic_data, metadata)
        cont_sim_mock.compute_breakdown.assert_called_once_with(
            real_data, synthetic_data, metadata)
        assert report._overall_quality_score == 0.14166666666666666
        assert report._property_breakdown == {
            'Column Shapes': 0.15000000000000002,
            'Column Pair Trends': 0.13333333333333333,
        }
        assert report._property_errors == {
            'Column Shapes': 0,
            'Column Pair Trends': 1,
        }

    @patch('sdmetrics.reports.single_table.quality_report.discretize_and_apply_metric')
    def test_generate_non_applicable_metric(self, mock_discretize_and_apply_metric):
        """Test the ``generate`` method with a non applicable metric.

        Expect that the single-table metrics are called. Expect that when one metric
        is not applicable and throws an ``IncomputableMetricError``, that metric is skipped.

        Setup:
        - Mock the expected single-table metric compute breakdown calls.
        - Mock one of the metrics to raise an ``IncomputableMetricError``

        Input:
        - Real data.
        - Synthetic data.
        - Metadata.

        Side Effects:
        - Expect that each single table metric's ``compute_breakdown`` methods are called once.
        - Expect that the ``_overall_quality_score`` and ``_property_breakdown`` attributes
          are populated.
        """
        # Setup
        real_data = pd.DataFrame({'col1': [1, 2, 3], 'col2': ['a', 'b', 'c']})
        synthetic_data = pd.DataFrame({'col1': [2, 2, 3], 'col2': ['b', 'a', 'c']})
        ks_complement_mock = Mock()
        metadata = {
            'columns': {
                'col1': {'sdtype': 'numerical'},
                'col2': {'sdtype': 'categorical'}
            }
        }
        ks_complement_mock.__name__ = 'KSComplement'
        ks_complement_mock.compute_breakdown.return_value = {
            'col1': {'score': 0.1},
            'col2': {'score': 0.2},
        }

        tv_complement_mock = Mock()
        tv_complement_mock.__name__ = 'TVComplement'
        tv_complement_mock.compute_breakdown.side_effect = IncomputableMetricError()

        corr_sim_mock = Mock()
        corr_sim_mock.__name__ = 'CorrelationSimilarity'
        corr_sim_mock.compute_breakdown.return_value = {
            ('col1', 'col2'): {'score': 0.1},
            ('col2', 'col3'): {'score': 0.2},
        }

        cont_sim_mock = Mock()
        cont_sim_mock.__name__ = 'ContingencySimilarity'
        cont_sim_mock.compute_breakdown.return_value = {
            ('col1', 'col2'): {'score': 0.1},
            ('col2', 'col3'): {'score': 0.2},
        }
        metrics_mock = {
            'Column Shapes': [ks_complement_mock, tv_complement_mock],
            'Column Pair Trends': [corr_sim_mock, cont_sim_mock],
        }
        mock_discretize_and_apply_metric.return_value = {}

        # Run
        with patch.object(
            QualityReport,
            'METRICS',
            metrics_mock,
        ):
            report = QualityReport()
            report.generate(real_data, synthetic_data, metadata)

        # Assert
        ks_complement_mock.compute_breakdown.assert_called_once_with(
            real_data, synthetic_data, metadata)
        tv_complement_mock.compute_breakdown.assert_called_once_with(
            real_data, synthetic_data, metadata)
        corr_sim_mock.compute_breakdown.assert_called_once_with(
            real_data, synthetic_data, metadata)
        cont_sim_mock.compute_breakdown.assert_called_once_with(
            real_data, synthetic_data, metadata)
        assert report._overall_quality_score == 0.15000000000000002
        assert report._property_breakdown == {
            'Column Shapes': 0.15000000000000002,
            'Column Pair Trends': 0.15000000000000002,
        }
        assert report._metric_results['TVComplement'] == {}

    def test_get_score(self):
        """Test the ``get_score`` method.

        Expect that the overall quality score is returned.

        Setup:
        - Mock the ``_overall_quality_score`` attribute.

        Input:
        - None

        Output:
        - The overall quality score.
        """
        # Setup
        report = QualityReport()
        mock_score = Mock()
        report._overall_quality_score = mock_score

        # Run
        score = report.get_score()

        # Assert
        assert score == mock_score

    def test_get_properties(self):
        """Test the ``get_details`` method.

        Expect that the property score breakdown is returned.

        Setup:
        - Mock the ``_property_breakdown`` attribute.

        Input:
        - None

        Output:
        - The metric scores for each property.
        """
        # Setup
        report = QualityReport()
        mock_property_breakdown = {'Column Shapes': 0.1, 'Column Pair Trends': 0.2}
        report._property_breakdown = mock_property_breakdown

        # Run
        properties = report.get_properties()

        # Assert
        pd.testing.assert_frame_equal(
            properties,
            pd.DataFrame({
                'Property': ['Column Shapes', 'Column Pair Trends'],
                'Score': [0.1, 0.2],
            }),
        )

    @patch('sdmetrics.reports.single_table.quality_report.pkg_resources.get_distribution')
    @patch('sdmetrics.reports.single_table.quality_report.pickle')
    def test_save(self, pickle_mock, get_distribution_mock):
        """Test the ``save`` method.

        Expect that the instance is passed to pickle.

        Input:
        - filepath

        Side Effects:
        - ``pickle`` is called with the instance.
        """
        # Setup
        report = Mock()
        open_mock = mock_open(read_data=pickle.dumps('test'))

        # Run
        with patch('sdmetrics.reports.single_table.quality_report.open', open_mock):
            QualityReport.save(report, 'test-file.pkl')

        # Assert
        get_distribution_mock.assert_called_once_with('sdmetrics')
        open_mock.assert_called_once_with('test-file.pkl', 'wb')
        pickle_mock.dump.assert_called_once_with(report, open_mock())
        assert report._package_version == get_distribution_mock.return_value.version

    @patch('sdmetrics.reports.single_table.quality_report.pkg_resources.get_distribution')
    @patch('sdmetrics.reports.single_table.quality_report.pickle')
    def test_load(self, pickle_mock, get_distribution_mock):
        """Test the ``load`` method.

        Expect that the report's load method is called with the expected args.

        Input:
        - filepath

        Output:
        - the loaded model

        Side Effects:
        - Expect that ``pickle`` is called with the filepath.
        """
        # Setup
        open_mock = mock_open(read_data=pickle.dumps('test'))
        report = Mock()
        pickle_mock.load.return_value = report
        report._package_version = get_distribution_mock.return_value.version

        # Run
        with patch('sdmetrics.reports.single_table.quality_report.open', open_mock):
            loaded = QualityReport.load('test-file.pkl')

        # Assert
        open_mock.assert_called_once_with('test-file.pkl', 'rb')
        assert loaded == pickle_mock.load.return_value

    @patch('sdmetrics.reports.single_table.quality_report.warnings')
    @patch('sdmetrics.reports.single_table.quality_report.pkg_resources.get_distribution')
    @patch('sdmetrics.reports.single_table.quality_report.pickle')
    def test_load_mismatched_versions(self, pickle_mock, get_distribution_mock, warnings_mock):
        """Test the ``load`` method with mismatched sdmetrics versions.

        Expect that the report's load method is called with the expected args.

        Input:
        - filepath

        Output:
        - the loaded model

        Side Effects:
        - Expect that ``pickle`` is called with the filepath.
        """
        # Setup
        open_mock = mock_open(read_data=pickle.dumps('test'))
        report = Mock()
        pickle_mock.load.return_value = report
        report._package_version = 'previous_version'
        get_distribution_mock.return_value.version = 'new_version'

        # Run
        with patch('sdmetrics.reports.single_table.quality_report.open', open_mock):
            loaded = QualityReport.load('test-file.pkl')

        # Assert
        open_mock.assert_called_once_with('test-file.pkl', 'rb')
        warnings_mock.warn.assert_called_once_with(
            'The report was created using SDMetrics version `previous_version` but you are '
            'currently using version `new_version`. Some features may not work as intended.'
        )
        assert loaded == pickle_mock.load.return_value

    @patch('sdmetrics.reports.single_table.quality_report.get_column_shapes_plot')
    def test_get_visualization_column_shapes(self, get_plot_mock):
        """Test the ``get_visualization`` method with Column Shapes.

        Input:
        - property='Column Shapes'

        Output:
        - the visualization

        Side Effects:
        - get_column_shapes_plot is called with the expected score breakdowns.
        """
        # Setup
        report = QualityReport()
        report._metric_results['KSComplement'] = {'score': 'ks_complement_score'}
        report._metric_results['TVComplement'] = {'score': 'tv_complement_score'}
        report._property_breakdown['Column Shapes'] = 0.78

        # Run
        fig = report.get_visualization('Column Shapes')

        # Assert
        get_plot_mock.assert_called_once_with({
            'KSComplement': {'score': 'ks_complement_score'},
            'TVComplement': {'score': 'tv_complement_score'},
        }, 0.78)
        assert fig == get_plot_mock.return_value

    @patch('sdmetrics.reports.single_table.quality_report.get_column_pairs_plot')
    def test_get_visualization_column_pairs(self, get_plot_mock):
        """Test the ``get_visualization`` method with Column Pairs.

        Input:
        - property='Column Pair Trends'

        Output:
        - the visualization

        Side Effects:
        - get_column_pairs_plot is called with the expected score breakdowns.
        """
        # Setup
        report = QualityReport()
        report._metric_results['CorrelationSimilarity'] = {'score': 'test_score_1'}
        report._metric_results['ContingencySimilarity'] = {'score': 'test_score_2'}
        report._property_breakdown['Column Pair Trends'] = 0.78

        # Run
        fig = report.get_visualization('Column Pair Trends')

        # Assert
        get_plot_mock.assert_called_once_with({
            'CorrelationSimilarity': {'score': 'test_score_1'},
            'ContingencySimilarity': {'score': 'test_score_2'},
        }, 0.78)
        assert fig == get_plot_mock.return_value

    def test_get_details(self):
        """Test the ``get_details`` method.

        Expect that the details of the desired property is returned.

        Input:
        - property name

        Output:
        - score details for the desired property
        """
        # Setup
        report = QualityReport()
        report._metric_results = {
            'KSComplement': {
                'col1': {'score': 0.1},
                'col2': {'score': 0.2},
            },
            'TVComplement': {
                'col1': {'score': 0.3},
                'col2': {'score': 0.4},
            }
        }

        # Run
        out = report.get_details('Column Shapes')

        # Assert
        pd.testing.assert_frame_equal(
            out,
            pd.DataFrame({
                'Column': ['col1', 'col2', 'col1', 'col2'],
                'Metric': ['KSComplement', 'KSComplement', 'TVComplement', 'TVComplement'],
                'Quality Score': [0.1, 0.2, 0.3, 0.4],
            })
        )

    def test_get_details_column_pair_trends(self):
        """Test the ``get_details`` method with column pair trends.

        Expect that the details of the desired property is returned.

        Input:
        - property name

        Output:
        - score details for the desired property
        """
        # Setup
        report = QualityReport()
        report._metric_results = {
            'CorrelationSimilarity': {
                ('col1', 'col3'): {'score': 0.1, 'real': 0.1, 'synthetic': 0.1},
                ('col2', 'col4'): {'score': 0.2, 'real': 0.2, 'synthetic': 0.2},
            },
            'ContingencySimilarity': {
                ('col1', 'col3'): {'score': 0.3, 'real': 0.3, 'synthetic': 0.3},
                ('col2', 'col4'): {'score': 0.4, 'real': 0.4, 'synthetic': 0.4},
            }
        }

        # Run
        out = report.get_details('Column Pair Trends')

        # Assert
        pd.testing.assert_frame_equal(
            out,
            pd.DataFrame({
                'Column 1': ['col1', 'col2', 'col1', 'col2'],
                'Column 2': ['col3', 'col4', 'col3', 'col4'],
                'Metric': [
                    'CorrelationSimilarity',
                    'CorrelationSimilarity',
                    'ContingencySimilarity',
                    'ContingencySimilarity',
                ],
                'Quality Score': [0.1, 0.2, 0.3, 0.4],
                'Real Correlation': [0.1, 0.2, 0.3, 0.4],
                'Synthetic Correlation': [0.1, 0.2, 0.3, 0.4],
            })
        )

    def test_get_details_column_pair_trends_with_errors(self):
        """Test the ``get_details`` method with column pair trends with errors.

        Expect that the details of the desired property is returned. Expect that the
        details result has an Error column.

        Input:
        - property name

        Output:
        - score details for the desired property
        """
        # Setup
        report = QualityReport()
        report._metric_results = {
            'CorrelationSimilarity': {
                ('col1', 'col3'): {'score': 0.1, 'real': 0.1, 'synthetic': 0.1},
                ('col2', 'col4'): {'score': 0.2, 'real': 0.2, 'synthetic': 0.2},
            },
            'ContingencySimilarity': {
                ('col1', 'col3'): {'score': 0.3, 'real': 0.3, 'synthetic': 0.3},
                ('col2', 'col4'): {'error': 'test error'},
            }
        }

        # Run
        out = report.get_details('Column Pair Trends')

        # Assert
        pd.testing.assert_frame_equal(
            out,
            pd.DataFrame({
                'Column 1': ['col1', 'col2', 'col1', 'col2'],
                'Column 2': ['col3', 'col4', 'col3', 'col4'],
                'Metric': [
                    'CorrelationSimilarity',
                    'CorrelationSimilarity',
                    'ContingencySimilarity',
                    'ContingencySimilarity',
                ],
                'Quality Score': [0.1, 0.2, 0.3, np.nan],
                'Real Correlation': [0.1, 0.2, 0.3, np.nan],
                'Synthetic Correlation': [0.1, 0.2, 0.3, np.nan],
                'Error': [np.nan, np.nan, np.nan, 'test error'],
            })
        )

    def test_get_raw_result(self):
        """Test the ``get_raw_result`` method.

        Expect that the raw result of the desired metric is returned. Expect that null
        scores are excluded.

        Input:
        - metric name

        Output:
        - Metric details
        """
        # Setup
        report = QualityReport()
        report._metric_results = {
            'KSComplement': {
                'col1': {'score': 0.1},
                'col2': {'score': 0.2},
                'col3': {'score': np.nan},
            },
        }

        # Run
        out = report.get_raw_result('KSComplement')

        # Assert
        assert out == [
            {
                'metric': {
                    'method': 'sdmetrics.single_table.multi_single_column.KSComplement',
                    'parameters': {},
                },
                'results': {
                    'col1': {'score': 0.1},
                    'col2': {'score': 0.2},
                }
            }
        ]

    def test__print_result(self):
        """Test the ``_print_results`` method.

        Expect that the correct messages are written.

        Input:
        - out argument

        Side Effects:
        - messages are written to the output.
        """
        # Setup
        report = QualityReport()
        report._overall_quality_score = 0.7
        report._property_breakdown = {'Column Shapes': 0.6, 'Column Pair Trends': 0.8}
        report._property_errors = {'Column Shapes': 0, 'Column Pair Trends': 0}
        mock_out = Mock()

        # Run
        report._print_results(mock_out)

        # Assert
        mock_out.write.assert_has_calls([
            call('\nOverall Quality Score: 70.0%\n\n'),
            call('Properties:\n'),
            call('Column Shapes: 60.0%\n'),
            call('Column Pair Trends: 80.0%\n'),
        ])

    def test__print_result_with_error(self):
        """Test the ``_print_results`` method with errors.

        Expect that the correct messages are written.

        Input:
        - out argument

        Side Effects:
        - messages are written to the output.
        """
        # Setup
        report = QualityReport()
        report._overall_quality_score = 0.6
        report._property_breakdown = {'Column Shapes': 0.6, 'Column Pair Trends': np.nan}
        report._property_errors = {'Column Shapes': 0, 'Column Pair Trends': 1}
        mock_out = Mock()

        # Run
        report._print_results(mock_out)

        # Assert
        mock_out.write.assert_has_calls([
            call('\nOverall Quality Score: 60.0%\n\n'),
            call('Properties:\n'),
            call('Column Shapes: 60.0%\n'),
            call('Column Pair Trends: Error computing property.\n'),
        ])

    def test__print_result_with_all_errors(self):
        """Test the ``_print_results`` method with all properties erroring out.

        Expect that the correct messages are written.

        Input:
        - out argument

        Side Effects:
        - messages are written to the output.
        """
        # Setup
        report = QualityReport()
        report._overall_quality_score = np.nan
        report._property_breakdown = {'Column Shapes': np.nan, 'Column Pair Trends': np.nan}
        report._property_errors = {'Column Shapes': 1, 'Column Pair Trends': 1}
        mock_out = Mock()

        # Run
        report._print_results(mock_out)

        # Assert
        mock_out.write.assert_has_calls([
            call('\nOverall Quality Score: Error computing report.\n\n'),
            call('Properties:\n'),
            call('Column Shapes: Error computing property.\n'),
            call('Column Pair Trends: Error computing property.\n'),
        ])
