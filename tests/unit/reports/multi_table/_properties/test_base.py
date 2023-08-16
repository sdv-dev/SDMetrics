"""Test BaseMultiTableProperty class."""

import re
from unittest.mock import Mock, patch

import pandas as pd
import pytest

from sdmetrics.reports.multi_table._properties import BaseMultiTableProperty


class TestBaseMultiTableProperty():

    def test__init__(self):
        """Test the ``__init__`` method."""
        # Setup
        base_property = BaseMultiTableProperty()

        # Assert
        assert base_property._properties == {}
        assert base_property._single_table_property is None
        assert base_property.is_computed is False

    def test__get_num_iterations(self):
        """Test that ``_get_num_iterations`` raises a ``NotImplementedError``."""
        # Setup
        base_property = BaseMultiTableProperty()

        # Run and Assert
        with pytest.raises(NotImplementedError):
            base_property._get_num_iterations(None)

    def test__generate_details_property(self):
        """Test the ``_generate_details`` method."""
        # Setup
        metadata = {
            'tables': {
                'Table_1': {},
                'Table_2': {},
            }
        }

        property_table_1 = Mock()
        property_table_1._details = pd.DataFrame({
            'Column': ['col1', 'col2'],
            'Score': [0.5, 0.6]
        })
        property_table_2 = Mock()
        property_table_2._details = pd.DataFrame({
            'Column': ['col3', 'col4'],
            'Score': [0.7, 0.8]
        })
        base_property = BaseMultiTableProperty()
        base_property._properties = {
            'Table_1': property_table_1,
            'Table_2': property_table_2,
        }
        base_property.details_property = pd.DataFrame()

        # Run
        base_property._generate_details(metadata)

        # Assert
        expected_details = pd.DataFrame({
            'Table': ['Table_1', 'Table_1', 'Table_2', 'Table_2'],
            'Column': ['col1', 'col2', 'col3', 'col4'],
            'Score': [0.5, 0.6, 0.7, 0.8]
        })

        pd.testing.assert_frame_equal(base_property.details_property, expected_details)

    def test__compute_average_raises_error(self):
        """Test that the method raises a ``ValueError`` when _details has not been computed."""
        # Setup
        base_property = BaseMultiTableProperty()

        # Run and Assert
        expected_error_message = re.escape(
            "The property details must be a DataFrame with a 'Score' column."
        )
        with pytest.raises(ValueError, match=expected_error_message):
            base_property._compute_average()

        base_property.details_property = pd.DataFrame({'Column': ['a', 'b', 'c']})
        with pytest.raises(ValueError, match=expected_error_message):
            base_property._compute_average()

    def test_get_score_raises_error(self):
        """Test that the method raises a ``NotImplementedError``."""
        # Setup
        base_property = BaseMultiTableProperty()

        # Run and Assert
        with pytest.raises(NotImplementedError):
            base_property.get_score(None, None, None, None)

    @patch('sdmetrics.reports.multi_table._properties.base.BaseMultiTableProperty'
           '._single_table_property', create=True)
    def test_get_score(self, mock_single_table_property):
        """Test the ``get_score`` method."""
        # Setup
        real_data = {
            'Table_1': pd.DataFrame(),
            'Table_2': pd.DataFrame(),
        }
        synthetic_data = {
            'Table_1': pd.DataFrame(),
            'Table_2': pd.DataFrame(),
        }
        metadata = {
            'tables': {
                'Table_1': {},
                'Table_2': {},
            }
        }
        progress_bar = 'tqdm'

        property_mock_1 = Mock()
        property_mock_2 = Mock()

        mock_single_table_property.side_effect = [property_mock_1, property_mock_2]

        base_multi_table_property = BaseMultiTableProperty()
        base_multi_table_property._generate_details = Mock()
        base_multi_table_property._compute_average = Mock(return_value=0.7)

        # Run
        result = base_multi_table_property.get_score(
            real_data, synthetic_data, metadata, progress_bar
        )

        # Assert
        property_mock_1.get_score.assert_called_once_with(
            real_data['Table_1'], synthetic_data['Table_1'], metadata['tables']['Table_1'], 'tqdm')
        property_mock_2.get_score.assert_called_once_with(
            real_data['Table_2'], synthetic_data['Table_2'], metadata['tables']['Table_2'], 'tqdm')
        base_multi_table_property._generate_details.assert_called_once_with(metadata)
        base_multi_table_property._compute_average.assert_called_once()
        assert result == 0.7
        assert base_multi_table_property.is_computed is True

    def test_get_visualization(self):
        """Test that the method returns the property's ``get_visualization``."""
        # Setup
        base_property = BaseMultiTableProperty()
        property_mock = Mock()
        base_property._properties = {'table': property_mock}
        base_property.is_computed = True

        # Run
        result = base_property.get_visualization('table')

        # Assert
        assert result == property_mock.get_visualization.return_value

    def test_get_visualization_raises_error(self):
        """Test that the method raises a ``ValueError`` when the table is not in the metadata."""
        # Setup
        base_property = BaseMultiTableProperty()

        # Run and Assert
        expected_message = (
            'The property must be computed before getting a visualization.'
            'Please call the ``get_score`` method first.'
        )
        with pytest.raises(ValueError, match=expected_message):
            base_property.get_visualization('table')
