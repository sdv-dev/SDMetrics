import pickle
import re
from unittest.mock import Mock, mock_open, patch

import numpy as np
import pandas as pd
import pytest

from sdmetrics.reports.multi_table import QualityReport
from sdmetrics.reports.multi_table._properties.cardinality import Cardinality
from sdmetrics.reports.multi_table._properties.column_pair_trends import ColumnPairTrends
from sdmetrics.reports.multi_table._properties.column_shapes import ColumnShapes


class TestQualityReport:

    def test___init__(self):
        """Test the ``__init__`` method.

        Expect that the correct attributes are initialized.
        """
        # Run
        report = QualityReport()

        # Assert
        assert report._tables == []
        assert report._overall_quality_score is None
        assert report._properties_instances == {}
        assert report._properties_scores == {}
        assert report._is_generated is False
        assert report._package_version is None

    @patch(
        'sdmetrics.reports.multi_table._properties.column_pair_trends.ColumnPairTrends.get_score',
        return_value=.2
    )
    @patch(
        'sdmetrics.reports.multi_table._properties.column_shapes.ColumnShapes.get_score',
        return_value=.4
    )
    @patch(
        'sdmetrics.reports.multi_table._properties.cardinality.Cardinality.get_score',
        return_value=.9
    )
    def test_generate(self, mock_cardinality_score, mock_column_shapes_score,
                      mock_column_pair_trends_score):
        """Test the ``generate`` method."""
        # Setup
        real_data = {
            'table1': pd.DataFrame({'id': [1, 2], 'col': [2, np.nan]}),
            'table2': pd.DataFrame({'id': [1, 2], 'col': ['a', np.nan]})
        }
        synth_data = {
            'table1': pd.DataFrame({'id': [1, 2], 'col': [3, np.nan]}),
            'table2': pd.DataFrame({'id': [1, 2], 'col': ['a', np.nan]})
        }
        metadata = {
            'tables': {
                'table1': {'columns': {'id': {'sdtype': 'id'}, 'col': {'sdtype': 'numerical'}}},
                'table2': {'columns': {'id': {'sdtype': 'id'}, 'col': {'sdtype': 'categorical'}}}
            },
            'relationships': [
                {
                    'parent_table_name': 'table1',
                    'parent_primary_key': 'id',
                    'child_table_name': 'table2',
                    'child_foreign_key': 'id'
                }
            ]
        }
        report = QualityReport()

        # Run
        report.generate(real_data, synth_data, metadata)

        # Assert
        assert isinstance(report._properties_instances['Column Shapes'], ColumnShapes)
        assert isinstance(report._properties_instances['Column Pair Trends'], ColumnPairTrends)
        assert isinstance(report._properties_instances['Cardinality'], Cardinality)

        assert report._properties_scores['Column Shapes'] == .4
        assert report._properties_scores['Column Pair Trends'] == .2
        assert report._properties_scores['Cardinality'] == .9

        assert report._overall_quality_score == .5
        assert report._is_generated is True

    def test_get_score(self):
        """Test the ``get_score`` method."""
        # Setup
        report = QualityReport()
        mock_score = Mock()
        report._overall_quality_score = mock_score
        report._is_generated = True

        # Run
        score = report.get_score()

        # Assert
        assert score == mock_score

    def test_get_score_not_generated(self):
        """Test the ``get_score`` method when the report hasn't been generated."""
        # Setup
        report = QualityReport()

        # Run and Assert
        msg = "The report has not been generated yet. Please call the 'generate' method."
        with pytest.raises(ValueError, match=msg):
            report.get_score()

    def test_get_properties(self):
        """Test the ``get_properties`` method."""
        # Setup
        report = QualityReport()
        report._properties_scores = {
            'Column Shapes': 0.1,
            'Column Pair Trends': 0.2,
            'Cardinality': 0.3,
        }
        report._is_generated = True

        # Run
        properties = report.get_properties()

        # Assert
        pd.testing.assert_frame_equal(
            properties,
            pd.DataFrame({
                'Property': ['Column Shapes', 'Column Pair Trends', 'Cardinality'],
                'Score': [0.1, 0.2, 0.3],
            }),
        )

    def test_get_properties_not_generated(self):
        """Test the ``get_properties`` method when the report hasn't been generated."""
        # Setup
        report = QualityReport()

        # Run and Assert
        msg = "The report has not been generated yet. Please call the 'generate' method."
        with pytest.raises(ValueError, match=msg):
            report.get_properties()

    def test_get_visualization(self):
        """Test the ``get_vizualization`` method."""
        # Setup
        report = QualityReport()
        instance = Mock()
        instance.get_visualization = Mock(return_value='visualization')
        report._properties_instances = {'Cardinality': instance}
        report._is_generated = True

        # Run
        visualization = report.get_visualization('Cardinality')

        # Assert
        instance.get_visualization.assert_called_once_with(None)
        assert visualization == 'visualization'

    def test_get_visualization_not_generated(self):
        """Test the ``get_visualization`` method when the report hasn't been generated."""
        # Setup
        report = QualityReport()

        # Run and Assert
        msg = "The report has not been generated yet. Please call the 'generate' method."
        with pytest.raises(ValueError, match=msg):
            report.get_visualization('property_name')

    def test_get_visualization_invalid_property(self):
        """Test it when the given property_name doesn't exist."""
        # Setup
        report = QualityReport()
        report._is_generated = True
        report._properties_instances = {'property_name': None}

        # Run and Assert
        msg = re.escape(
            "Invalid property name ('invalid_name'). It must be one of "
            "['property_name']."
        )
        with pytest.raises(ValueError, match=msg):
            report.get_visualization('invalid_name')

    def test_get_visualization_missing_table_name(self):
        """Test it when table_name is missing and property is not Cardinality."""
        # Setup
        report = QualityReport()
        report._is_generated = True
        report._properties_instances = {'Column Shapes': None}
        report._tables = ['tab1']

        # Run and Assert
        msg = "Table name must be provided when viewing details for property 'Column Shapes'."
        with pytest.raises(ValueError, match=msg):
            report.get_visualization('Column Shapes')

    def test_get_visualization_invalid_table_name(self):
        """Test it when table_name is invalid."""
        # Setup
        report = QualityReport()
        report._is_generated = True
        report._properties_instances = {'Column Shapes': None}
        report._tables = ['table']

        # Run and Assert
        msg = re.escape("Unknown table ('invalid_table'). Must be one of ['table'].")
        with pytest.raises(ValueError, match=msg):
            report.get_visualization('Column Shapes', 'invalid_table')

    def test_get_details(self):
        """Test the ``get_details`` method."""
        # Setup
        report = QualityReport()
        instance = Mock()
        instance._details = {'details'}
        report._properties_instances = {'Cardinality': instance}
        report._is_generated = True

        # Run
        details = report.get_details('Cardinality')

        # Assert
        assert details == {'details'}

    def test_get_details_table_name(self):
        """Test the ``get_details`` method with Cardinality and table_name."""
        # Setup
        report = QualityReport()
        instance = Mock()
        instance._details = {('table1', 'table2'): 'details1', ('table1', 'table3'): 'details2'}
        report._properties_instances = {'Cardinality': instance}
        report._is_generated = True
        report._tables = ['table1', 'table2', 'table3']

        # Run
        details = report.get_details('Cardinality', 'table3')

        # Assert
        assert details == {('table1', 'table3'): 'details2'}

    def test_get_details_no_table_name(self):
        """Test it works when table_name is None and property is not Cardinality."""
        # Setup
        report = QualityReport()
        report._is_generated = True
        instance = Mock()
        details_mock = Mock()
        details_mock._details = {'details'}
        instance._properties = {'table': details_mock}
        report._properties_instances = {'Column Shapes': instance}
        report._tables = ['tab1']

        # Run
        details = report.get_details('Column Shapes')

        # Assert
        assert details == {'table': {'details'}}

    def test_get_details_not_generated(self):
        """Test the ``get_details`` method when the report hasn't been generated."""
        # Setup
        report = QualityReport()

        # Run and Assert
        msg = "The report has not been generated yet. Please call the 'generate' method."
        with pytest.raises(ValueError, match=msg):
            report.get_details('property_name')

    def test_get_details_invalid_property(self):
        """Test it when the given property_name doesn't exist."""
        # Setup
        report = QualityReport()
        report._is_generated = True
        report._properties_instances = {'property_name': None}

        # Run and Assert
        msg = re.escape(
            "Invalid property name ('invalid_name'). It must be one of "
            "['property_name']."
        )
        with pytest.raises(ValueError, match=msg):
            report.get_details('invalid_name')

    def test_get_details_invalid_table_name(self):
        """Test it when table_name is invalid."""
        # Setup
        report = QualityReport()
        report._is_generated = True
        report._properties_instances = {'Column Shapes': None}
        report._tables = ['table']

        # Run and Assert
        msg = re.escape("Unknown table ('invalid_table'). Must be one of ['table'].")
        with pytest.raises(ValueError, match=msg):
            report.get_details('Column Shapes', 'invalid_table')

    @patch('sdmetrics.reports.multi_table.quality_report.pkg_resources.get_distribution')
    @patch('sdmetrics.reports.multi_table.quality_report.pickle')
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
        with patch('sdmetrics.reports.multi_table.quality_report.open', open_mock):
            QualityReport.save(report, 'test-file.pkl')

        # Assert
        get_distribution_mock.assert_called_once_with('sdmetrics')
        open_mock.assert_called_once_with('test-file.pkl', 'wb')
        pickle_mock.dump.assert_called_once_with(report, open_mock())
        assert report._package_version == get_distribution_mock.return_value.version

    @patch('sdmetrics.reports.multi_table.quality_report.pkg_resources.get_distribution')
    @patch('sdmetrics.reports.multi_table.quality_report.pickle')
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
        with patch('sdmetrics.reports.multi_table.quality_report.open', open_mock):
            loaded = QualityReport.load('test-file.pkl')

        # Assert
        open_mock.assert_called_once_with('test-file.pkl', 'rb')
        assert loaded == pickle_mock.load.return_value

    @patch('sdmetrics.reports.multi_table.quality_report.warnings')
    @patch('sdmetrics.reports.multi_table.quality_report.pkg_resources.get_distribution')
    @patch('sdmetrics.reports.multi_table.quality_report.pickle')
    def test_load_mismatched_versions(self, pickle_mock, get_distribution_mock, warnings_mock):
        """Test the ``load`` method with mismatched sdmetrics versions.

        Expect that the report's load method is called with the expected args.
        Expect that a warning is raised.

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
        with patch('sdmetrics.reports.multi_table.quality_report.open', open_mock):
            loaded = QualityReport.load('test-file.pkl')

        # Assert
        open_mock.assert_called_once_with('test-file.pkl', 'rb')
        warnings_mock.warn.assert_called_once_with(
            'The report was created using SDMetrics version `previous_version` but you are '
            'currently using version `new_version`. Some features may not work as intended.'
        )
        assert loaded == pickle_mock.load.return_value
