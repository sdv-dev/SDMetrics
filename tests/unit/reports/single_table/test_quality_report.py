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

    def test_generate(self):
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
        real_data = Mock()
        synthetic_data = Mock()
        ks_complement_mock = Mock()
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

        # Run
        with patch.object(
            QualityReport,
            'METRICS',
            metrics_mock,
        ):
            report = QualityReport()
            report.generate(real_data, synthetic_data, Mock())

        # Assert
        ks_complement_mock.compute_breakdown.assert_called_once_with(real_data, synthetic_data)
        tv_complement_mock.compute_breakdown.assert_called_once_with(real_data, synthetic_data)
        corr_sim_mock.compute_breakdown.assert_called_once_with(real_data, synthetic_data)
        cont_sim_mock.compute_breakdown.assert_called_once_with(real_data, synthetic_data)
        assert report._overall_quality_score == 0.15000000000000002
        assert report._property_breakdown == {
            'Column Shapes': 0.15000000000000002,
            'Column Pair Trends': 0.15000000000000002,
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

        # Run
        report.show_details('Column Shapes')

        # Assert
        get_plot_mock.assert_called_once_with({
            'KSComplement': {'score': 'ks_complement_score'},
            'TVComplement': {'score': 'tv_complement_score'},
        })

    @patch('sdmetrics.reports.single_table.quality_report.get_column_pairs_plot')
    def test_show_details_column_pairs(self, get_plot_mock):
        """Test the ``show_details`` method with Column Pairs.

        Input:
        - property='Column Pairs'

        Side Effects:
        - get_column_pairs_plot is called with the expected score breakdowns.
        """
        # Setup
        report = QualityReport()
        report._metric_results['CorrelationSimilarity'] = {'score': 'test_score_1'}
        report._metric_results['ContingencySimilarity'] = {'score': 'test_score_2'}
        mock_real_corr = Mock()
        report._real_corr = mock_real_corr
        mock_synth_corr = Mock()
        report._synth_corr = mock_synth_corr

        # Run
        report.show_details('Column Pairs')

        # Assert
        get_plot_mock.assert_called_once_with({
            'CorrelationSimilarity': {'score': 'test_score_1'},
            'ContingencySimilarity': {'score': 'test_score_2'},
        }, mock_real_corr, mock_synth_corr)
