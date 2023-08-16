import re
from unittest.mock import Mock, call, patch

import pandas as pd
import pytest

from sdmetrics.reports.multi_table.base_multi_table_report import BaseMultiTableReport


class TestBaseReport:

    def test__init__(self):
        """Test the ``__init__`` method."""
        # Setup
        report = BaseMultiTableReport()

        # Assert
        assert report._overall_score is None
        assert report.is_generated is False
        assert report._properties == {}
        assert report.table_names == []

    def test__validate_relationships(self):
        """Test the ``_validate_relationships`` method."""
        # Setup
        real_data = {
            'Table_1': pd.DataFrame({'col1': [1, 2, 3]}),
            'Table_2': pd.DataFrame({'col2': [1, 2, 3]}),
        }
        real_data_bad = {
            'Table_1': pd.DataFrame({'col1': ['1', '2', '3']}),
            'Table_2': pd.DataFrame({'col2': [1, 2, 3]}),
        }
        synthetic_data = {
            'Table_1': pd.DataFrame({'col1': [1, 2, 3]}),
            'Table_2': pd.DataFrame({'col2': [1, 2, 3]}),
        }
        metadata = {
            'tables': {
                'Table_1': {
                    'columns': {
                        'col1': {},
                    },
                },
                'Table_2': {
                    'columns': {
                        'col2': {}
                    },
                },
            },
            'relationships': [
                {
                    'parent_table_name': 'Table_1',
                    'parent_primary_key': 'col1',
                    'child_table_name': 'Table_2',
                    'child_foreign_key': 'col2'
                },
            ]
        }

        report = BaseMultiTableReport()

        # Run and Assert
        report._validate_relationships(real_data, synthetic_data, metadata)

        expected_error_message = re.escape(
            "The 'Table_1' table and 'Table_2' table cannot be merged. Please "
            "make sure the primary key in 'Table_1' ('col1') and the "
            "foreign key in 'Table_2' ('col2') have the same data type."
        )
        with pytest.raises(ValueError, match=expected_error_message):
            report._validate_metadata_matches_data(real_data_bad, synthetic_data, metadata)

    @patch('sdmetrics.reports.base_report.BaseReport._validate_metadata_matches_data')
    def test__validate_metadata_matches_data(self, mock__validate_metadata_matches_data):
        """Test the ``_validate_metadata_matches_data`` method."""
        # Setup
        real_data = {
            'Table_1': pd.DataFrame({'col1': [1, 2, 3]}),
            'Table_2': pd.DataFrame({'col2': [4, 5, 6]}),
        }
        synthetic_data = {
            'Table_1': pd.DataFrame({'col1': [1, 2, 3]}),
            'Table_2': pd.DataFrame({'col2': [4, 5, 6]}),
        }
        metadata = {
            'tables': {
                'Table_1': {
                    'columns': {
                        'col1': {},
                    },
                },
                'Table_2': {
                    'columns': {
                        'col2': {}
                    },
                },
            },
            'relationships': [
                {
                    'parent_table_name': 'Table_1',
                    'parent_primary_key': 'col1',
                    'child_table_name': 'Table_2',
                    'child_foreign_key': 'col2'
                },
            ]
        }
        mock__validate_relationships = Mock()

        report = BaseMultiTableReport()
        report._validate_relationships = mock__validate_relationships

        # Run
        report._validate_metadata_matches_data(real_data, synthetic_data, metadata)

        # Assert
        expected_calls = [
            call(real_data['Table_1'], synthetic_data['Table_1'], metadata['tables']['Table_1']),
            call(real_data['Table_2'], synthetic_data['Table_2'], metadata['tables']['Table_2']),
        ]
        mock__validate_metadata_matches_data.assert_has_calls(expected_calls)
        report._validate_relationships.assert_called_once_with(real_data, synthetic_data, metadata)

    def test__check_table_names(self):
        """Test the ``_check_table_names`` method."""
        # Setup
        report = BaseMultiTableReport()
        report.table_names = ['Table_1', 'Table_2']

        # Run and Assert
        report._check_table_names('Table_1')

        expected_error_message = re.escape(
            "Unknown table ('Table_3'). Must be one of ['Table_1', 'Table_2']."
        )

        with pytest.raises(ValueError, match=expected_error_message):
            report._check_table_names('Table_3')

    def test_get_details(self):
        """Test the ``get_details`` method."""
        # Setup
        details_property_df = pd.DataFrame({
            'Table': ['Table_1', 'Table_1', 'Table_2'],
            'Column': ['col1', 'col2', 'col3'],
            'Score': [0.3, 0.4, 0.5],
        })

        mock__validate_property_generated = Mock()

        report = BaseMultiTableReport()
        report._validate_property_generated = mock__validate_property_generated

        property_1 = Mock()
        property_1.details = details_property_df

        report._properties = {
            'Property_1': property_1,
            'Property_2': Mock(),
        }

        # Run
        result = report.get_details('Property_1')

        # Assert
        mock__validate_property_generated.assert_called_once_with('Property_1')
        pd.testing.assert_frame_equal(result, details_property_df)

    def test_get_details_with_table_name(self):
        """Test the ``get_details`` method when a table name is given."""
        # Setup
        details_property_df = pd.DataFrame({
            'Table': ['Table_1', 'Table_1', 'Table_2'],
            'Column': ['col1', 'col2', 'col3'],
            'Score': [0.3, 0.4, 0.5],
        })

        mock__check_table_names = Mock()
        mock__validate_property_generated = Mock()

        report = BaseMultiTableReport()
        report._check_table_names = mock__check_table_names
        report._validate_property_generated = mock__validate_property_generated

        expected_details = pd.DataFrame({
            'Column': ['col1', 'col2'],
            'Score': [0.3, 0.4],
        })

        property_1 = Mock()
        property_1.details = details_property_df
        property_1.get_details = Mock(return_value=expected_details)
        report._properties = {
            'Property_1': property_1,
            'Property_2': Mock(),
        }

        # Run
        result = report.get_details('Property_1', 'Table_1')

        # Assert
        mock__check_table_names.assert_called_once_with('Table_1')
        mock__validate_property_generated.assert_called_once_with('Property_1')
        property_1.get_details.assert_called_once_with('Table_1')

        pd.testing.assert_frame_equal(result, expected_details)

    def test_get_visualization_with_table_name(self):
        """Test the ``get_visualization`` method when a table name is given."""
        # Setup
        mock_visualization = 'Visualization_1'
        mock__validate_property_generated = Mock()
        mock__check_table_names = Mock()
        mock_get_visualization = Mock(return_value=mock_visualization)

        report = BaseMultiTableReport()
        report._validate_property_generated = mock__validate_property_generated
        report._check_table_names = mock__check_table_names

        mock_property = Mock(get_visualization=mock_get_visualization)

        report._properties = {
            'Property_1': mock_property,
            'Property_2': Mock(),
        }

        # Run
        result = report.get_visualization('Property_1', 'Table_1')

        # Assert
        mock__validate_property_generated.assert_called_once_with('Property_1')
        mock__check_table_names.assert_called_once_with('Table_1')
        mock_get_visualization.assert_called_once_with('Table_1')

        assert result == mock_visualization

    def test_get_visualization_without_table_name(self):
        """Test the ``get_visualization`` method when a table name is not given."""
        # Setup
        report = BaseMultiTableReport()

        # Assert
        expected_error_message = re.escape(
            'Please provide a table name to get a visualization for the property.'
        )

        with pytest.raises(ValueError, match=expected_error_message):
            report.get_visualization('Property_1')
