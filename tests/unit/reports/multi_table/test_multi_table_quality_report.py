import pickle
from unittest.mock import Mock, mock_open, patch

import pandas as pd
import pytest

from sdmetrics.reports.multi_table import QualityReport


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

        Expect that the multi-table metrics are called.

        Setup:
        - Mock the expected multi-table metric compute breakdown calls.

        Input:
        - Real data.
        - Synthetic data.
        - Metadata.

        Side Effects:
        - Expect that each multi table metric's ``compute_breakdown`` methods are called once.
        - Expect that the ``_overall_quality_score`` and ``_property_breakdown`` attributes
          are populated.
        """
        # Setup
        real_data = Mock()
        synthetic_data = Mock()
        metadata = Mock()

        ks_complement_mock = Mock()
        ks_complement_mock.__name__ = 'KSComplement'
        ks_complement_mock.compute_breakdown.return_value = {
            'table1': {
                'col1': {'score': 0.1},
                'col2': {'score': 0.2},
            }
        }

        tv_complement_mock = Mock()
        tv_complement_mock.__name__ = 'TVComplement'
        tv_complement_mock.compute_breakdown.return_value = {
            'table1': {
                'col1': {'score': 0.1},
                'col2': {'score': 0.2},
            }
        }

        corr_sim_mock = Mock()
        corr_sim_mock.__name__ = 'CorrelationSimilarity'
        corr_sim_mock.compute_breakdown.return_value = {
            'table1': {
                'col1': {'score': 0.1},
                'col2': {'score': 0.2},
            }
        }

        cont_sim_mock = Mock()
        cont_sim_mock.__name__ = 'ContingencySimilarity'
        cont_sim_mock.compute_breakdown.return_value = {
            'table1': {
                'col1': {'score': 0.1},
                'col2': {'score': 0.2},
            }
        }

        cardinality_mock = Mock()
        cardinality_mock.__name__ = 'CardinalityShapeSimilarity'
        cardinality_mock.compute_breakdown.return_value = {
            ('table1', 'table2'): {'score': 1.0},
        }
        metrics_mock = {
            'Column Shapes': [ks_complement_mock, tv_complement_mock],
            'Column Pair Trends': [corr_sim_mock, cont_sim_mock],
            'Parent Child Relationships': [cardinality_mock],
        }

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
        cardinality_mock.compute_breakdown.assert_called_once_with(
            real_data, synthetic_data, metadata)

        assert report._overall_quality_score == 0.43333333333333335
        assert report._property_breakdown == {
            'Column Shapes': 0.15000000000000002,
            'Column Pair Trends': 0.15000000000000002,
            'Parent Child Relationships': 1.0,
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
        mock_property_breakdown = {
            'Column Shapes': 0.1,
            'Column Pair Trends': 0.2,
            'Parent Child Relationships': 0.3,
        }
        report._property_breakdown = mock_property_breakdown

        # Run
        properties = report.get_properties()

        # Assert
        pd.testing.assert_frame_equal(
            properties,
            pd.DataFrame({
                'Property': ['Column Shapes', 'Column Pair Trends', 'Parent Child Relationships'],
                'Score': [0.1, 0.2, 0.3],
            }),
        )

    @patch('sdmetrics.reports.multi_table.quality_report.pickle')
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
        with patch('sdmetrics.reports.multi_table.quality_report.open', open_mock):
            QualityReport.save(report, 'test-file.pkl')

        # Assert
        open_mock.assert_called_once_with('test-file.pkl', 'wb')
        pickle_mock.dump.assert_called_once_with(report, open_mock())

    @patch('sdmetrics.reports.multi_table.quality_report.pickle')
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
        with patch('sdmetrics.reports.multi_table.quality_report.open', open_mock):
            loaded = QualityReport.load('test-file.pkl')

        # Assert
        open_mock.assert_called_once_with('test-file.pkl', 'rb')
        assert loaded == pickle_mock.load.return_value

    @patch('sdmetrics.reports.multi_table.quality_report.get_column_shapes_plot')
    def test_show_details_column_shapes(self, get_plot_mock):
        """Test the ``show_details`` method with Column Shapes.

        Input:
        - property='Column Shapes'

        Side Effects:
        - get_column_shapes_plot is called with the expected score breakdowns.
        """
        # Setup
        report = QualityReport()
        report._metric_results['KSComplement'] = {
            'table1': {'col1': {'score': 'ks_complement_score'}},
            'table2': {'col3': {'score': 'other_score'}},
        }
        report._metric_results['TVComplement'] = {
            'table1': {'col2': {'score': 'tv_complement_score'}},
            'table2': {'col4': {'score': 'other_score'}},
        }

        # Run
        report.show_details('Column Shapes', table_name='table1')

        # Assert
        get_plot_mock.assert_called_once_with({
            'KSComplement': {'col1': {'score': 'ks_complement_score'}},
            'TVComplement': {'col2': {'score': 'tv_complement_score'}},
        })

    def test_show_details_column_shapes_no_table_name(self):
        """Test the ``show_details`` method with Column Shapes and no table name.

        Expect that a ``ValueError`` is thrown.

        Input:
        - property='Column Shapes'
        - no table_name

        Side Effects:
        - a ``ValueError`` is thrown.
        """
        # Setup
        report = QualityReport()

        # Run and assert
        with pytest.raises(
            ValueError,
            match='Table name must be provided when viewing details for property Column Shapes',
        ):
            report.show_details('Column Shapes')

    @patch('sdmetrics.reports.multi_table.quality_report.get_column_pairs_plot')
    def test_show_details_column_pairs(self, get_plot_mock):
        """Test the ``show_details`` method with Column Pairs.

        Input:
        - property='Column Pairs'

        Side Effects:
        - get_column_pairs_plot is called with the expected score breakdowns.
        """
        # Setup
        report = QualityReport()
        report._metric_results['CorrelationSimilarity'] = {
            'table1': {('col1', 'col2'): {'score': 'test_score_1'}},
            'table2': {('col3', 'col4'): {'score': 'other_score'}},
        }
        report._metric_results['ContingencySimilarity'] = {
            'table1': {('col5', 'col6'): {'score': 'test_score_2'}},
            'table2': {('col7', 'col8'): {'score': 'other_score'}},
        }

        mock_real_corr = Mock()
        report._real_corr = mock_real_corr
        mock_synth_corr = Mock()
        report._synth_corr = mock_synth_corr

        # Run
        report.show_details('Column Pair Trends', table_name='table1')

        # Assert
        get_plot_mock.assert_called_once_with({
            'CorrelationSimilarity': {('col1', 'col2'): {'score': 'test_score_1'}},
            'ContingencySimilarity': {('col5', 'col6'): {'score': 'test_score_2'}},
        }, mock_real_corr, mock_synth_corr)

    @patch('sdmetrics.reports.multi_table.quality_report.get_table_relationships_plot')
    def test_show_details_table_relationships(self, get_plot_mock):
        """Test the ``show_details`` method with Parent Child Relationships.

        Input:
        - property='Parent Child Relationships'

        Side Effects:
        - get_parent_child_relationships_plot is called with the expected score breakdowns.
        """
        # Setup
        report = QualityReport()
        report._metric_results['CardinalityShapeSimilarity'] = {
            ('table1', 'table2'): {'score': 'test_score_1'},
            ('table3', 'table2'): {'score': 'test_score_2'},
        }

        # Run
        report.show_details('Parent Child Relationships')

        # Assert
        get_plot_mock.assert_called_once_with({
            'CardinalityShapeSimilarity': {
                ('table1', 'table2'): {'score': 'test_score_1'},
                ('table3', 'table2'): {'score': 'test_score_2'},
            },
        })