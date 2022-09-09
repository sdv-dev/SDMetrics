import pickle
from unittest.mock import Mock, mock_open, patch

import pandas as pd

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

    @patch('sdmetrics.reports.multi_table.quality_report.discretize_and_apply_metric')
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
        metadata = {'fields': {'col1': {'type': 'numerical'}, 'col2': {'type': 'categorical'}}}
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
            'col1': {'score': 0.1},
            'col2': {'score': 0.2},
        }

        cont_sim_mock = Mock()
        cont_sim_mock.__name__ = 'ContingencySimilarity'
        cont_sim_mock.compute_breakdown.return_value = {
            'col1': {'score': 0.1},
            'col2': {'score': 0.2},
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
        assert report._overall_quality_score == 0.1930555555555556
        assert report._property_breakdown == {
            'Column Shapes': 0.15000000000000002,
            'Column Pair Trends': 0.23611111111111113,
        }

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

    @patch('sdmetrics.reports.single_table.quality_report.pickle')
    def test_save(self, pickle_mock):
        """Test the ``save`` method.

        Expect that the instance is passed to pickle.

        Input:
        - filename

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
        open_mock.assert_called_once_with('test-file.pkl', 'wb')
        pickle_mock.dump.assert_called_once_with(report, open_mock())

    @patch('sdmetrics.reports.single_table.quality_report.pickle')
    def test_load(self, pickle_mock):
        """Test the ``load`` method.

        Expect that the report's load method is called with the expected args.

        Input:
        - filename

        Output:
        - the loaded model

        Side Effects:
        - Expect that ``pickle`` is called with the filename.
        """
        # Setup
        open_mock = mock_open(read_data=pickle.dumps('test'))

        # Run
        with patch('sdmetrics.reports.single_table.quality_report.open', open_mock):
            loaded = QualityReport.load('test-file.pkl')

        # Assert
        open_mock.assert_called_once_with('test-file.pkl', 'rb')
        assert loaded == pickle_mock.load.return_value

    @patch('sdmetrics.reports.single_table.quality_report.get_column_shapes_plot')
    def test_show_details_column_shapes(self, get_plot_mock):
        """Test the ``show_details`` method with Column Shapes.

        Input:
        - property='Column Shapes'

        Side Effects:
        - get_column_shapes_plot is called with the expected score breakdowns.
        """
        # Setup
        report = QualityReport()
        report._metric_results['KSComplement'] = {'score': 'ks_complement_score'}
        report._metric_results['TVComplement'] = {'score': 'tv_complement_score'}
        report._property_breakdown['Column Shapes'] = 0.78

        # Run
        report.show_details('Column Shapes')

        # Assert
        get_plot_mock.assert_called_once_with({
            'KSComplement': {'score': 'ks_complement_score'},
            'TVComplement': {'score': 'tv_complement_score'},
        }, 0.78)

    @patch('sdmetrics.reports.single_table.quality_report.get_column_pairs_plot')
    def test_show_details_column_pairs(self, get_plot_mock):
        """Test the ``show_details`` method with Column Pairs.

        Input:
        - property='Column Pair Trends'

        Side Effects:
        - get_column_pairs_plot is called with the expected score breakdowns.
        """
        # Setup
        report = QualityReport()
        report._metric_results['CorrelationSimilarity'] = {'score': 'test_score_1'}
        report._metric_results['ContingencySimilarity'] = {'score': 'test_score_2'}
        report._property_breakdown['Column Pair Trends'] = 0.78

        # Run
        report.show_details('Column Pair Trends')

        # Assert
        get_plot_mock.assert_called_once_with({
            'CorrelationSimilarity': {'score': 'test_score_1'},
            'ContingencySimilarity': {'score': 'test_score_2'},
        }, 0.78)

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
                'Columns': [
                    ('col1', 'col3'),
                    ('col2', 'col4'),
                    ('col1', 'col3'),
                    ('col2', 'col4'),
                ],
                'Metric': [
                    'CorrelationSimilarity',
                    'CorrelationSimilarity',
                    'ContingencySimilarity',
                    'ContingencySimilarity',
                ],
                'Quality Score': [0.1, 0.2, 0.3, 0.4],
                'Real Score': [0.1, 0.2, 0.3, 0.4],
                'Synthetic Score': [0.1, 0.2, 0.3, 0.4],
            })
        )

    def test_get_raw_result(self):
        """Test the ``get_raw_result`` method.

        Expect that the raw result of the desired metric is returned.

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
