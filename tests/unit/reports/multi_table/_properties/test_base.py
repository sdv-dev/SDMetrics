"""Test BaseMultiTableProperty class."""

from unittest.mock import Mock

import numpy as np
import pandas as pd
import pytest

from sdmetrics.reports.multi_table._properties import BaseMultiTableProperty
from tests.utils import DataFrameMatcher


class TestBaseMultiTableProperty():

    def test__init__(self):
        """Test the ``__init__`` method."""
        # Setup
        base_property = BaseMultiTableProperty()

        # Assert
        assert base_property._properties == {}
        assert base_property._single_table_property is None
        assert base_property._num_iteration_case is None
        assert base_property.is_computed is False

    def test__get_num_iterations(self):
        """Test that ``_get_num_iterations``."""
        # Setup
        base_property = BaseMultiTableProperty()

        metadata = {
            'tables': {
                'table1': {
                    'columns': {
                        'col1': {}, 'col2': {}, 'col3': {},
                        'col4': {}, 'col5': {},
                    }
                },
                'table2': {
                    'columns': {
                        'col6': {},
                    }
                },
                'table3': {
                    'columns': {
                        'col7': {},
                    }
                },
                'table4': {
                    'columns': {
                        'col8': {},
                    }
                },
            },
            'relationships': [
                {
                    'parent_table_name': 'table1',
                    'parent_primary_key': 'col1',
                    'child_table_name': 'table2',
                    'child_foreign_key': 'col6'
                },
                {
                    'parent_table_name': 'table1',
                    'parent_primary_key': 'col1',
                    'child_table_name': 'table3',
                    'child_foreign_key': 'col7'
                },
                {
                    'parent_table_name': 'table2',
                    'parent_primary_key': 'col6',
                    'child_table_name': 'table4',
                    'child_foreign_key': 'col8'
                },
            ]
        }

        # Run and Assert
        base_property._num_iteration_case = 'column'
        assert base_property._get_num_iterations(metadata) == 8

        base_property._num_iteration_case = 'table'
        assert base_property._get_num_iterations(metadata) == 4

        base_property._num_iteration_case = 'relationship'
        assert base_property._get_num_iterations(metadata) == 3

        base_property._num_iteration_case = 'column_pair'
        assert base_property._get_num_iterations(metadata) == 10

        base_property._num_iteration_case = 'inter_table_column_pair'
        assert base_property._get_num_iterations(metadata) == 11

    def test__generate_details_property(self):
        """Test the ``_generate_details`` method."""
        # Setup
        real_data = {
            'Table_1': pd.DataFrame({
                'col1': [0, 1],
                'col2': [0, 1]
            }),
            'Table_2': pd.DataFrame({
                'col3': [2, 3],
                'col4': [2, 3]
            }),
        }
        synthetic_data = {
            'Table_1': pd.DataFrame({
                'col1': [4, 5],
                'col2': [4, 5]
            }),
            'Table_2': pd.DataFrame({
                'col3': [6, 7],
                'col4': [6, 7]
            }),
        }

        metadata = {
            'tables': {
                'Table_1': {},
                'Table_2': {},
            }
        }

        progress_bar_mock = Mock()

        property_table_1 = Mock()
        property_table_1.details = pd.DataFrame({
            'Column': ['col1', 'col2'],
            'Score': [0.5, 0.6]
        })
        property_table_2 = Mock()
        property_table_2.details = pd.DataFrame({
            'Column': ['col3', 'col4'],
            'Score': [0.7, 0.8]
        })
        base_property = BaseMultiTableProperty()
        base_property._properties = {
            'Table_1': property_table_1,
            'Table_2': property_table_2,
        }
        base_property.details = pd.DataFrame()
        base_property._single_table_property = Mock()
        base_property._single_table_property.side_effect = [property_table_1, property_table_2]

        # Run
        base_property._generate_details(real_data, synthetic_data, metadata, progress_bar_mock)

        # Assert
        expected_details = pd.DataFrame({
            'Table': ['Table_1', 'Table_1', 'Table_2', 'Table_2'],
            'Column': ['col1', 'col2', 'col3', 'col4'],
            'Score': [0.5, 0.6, 0.7, 0.8]
        })
        property_table_1.get_score.assert_called_once_with(
            DataFrameMatcher(real_data['Table_1']),
            DataFrameMatcher(synthetic_data['Table_1']),
            metadata['tables']['Table_1'],
            progress_bar_mock
        )
        property_table_2.get_score.assert_called_once_with(
            DataFrameMatcher(real_data['Table_2']),
            DataFrameMatcher(synthetic_data['Table_2']),
            metadata['tables']['Table_2'],
            progress_bar_mock
        )

        pd.testing.assert_frame_equal(base_property.details, expected_details)

    def test__generate_details_raises_error(self):
        """Test that the method raises a ``NotImplementedError``."""
        # Setup
        base_property = BaseMultiTableProperty()

        # Run and Assert
        with pytest.raises(NotImplementedError):
            base_property._generate_details(None, None, None, None)

    def test__compute_average_sends_nan(self):
        """Test that the method raises an error when _details has not been computed."""
        # Setup
        base_property = BaseMultiTableProperty()

        # Run and Assert
        assert np.isnan(base_property._compute_average())
        base_property.details = pd.DataFrame({'Column': ['a', 'b', 'c']})
        assert np.isnan(base_property._compute_average())

    def test_get_score(self):
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

        base_multi_table_property = BaseMultiTableProperty()
        base_multi_table_property._generate_details = Mock()
        base_multi_table_property._compute_average = Mock(return_value=0.7)

        details_df = pd.DataFrame({
            'Table': ['Table1', 'Table2'],
            'Metric': ['Metric1', 'Metric2'],
            'Score': [1.0, 1.0],
            'Error': [None, None]
        })
        base_multi_table_property.details = details_df.copy()

        # Run
        result = base_multi_table_property.get_score(
            real_data, synthetic_data, metadata, progress_bar
        )

        # Assert
        base_multi_table_property._generate_details.assert_called_once_with(
            real_data, synthetic_data, metadata, progress_bar
        )
        base_multi_table_property._compute_average.assert_called_once()
        assert result == 0.7
        assert base_multi_table_property.is_computed is True
        pd.testing.assert_frame_equal(
            base_multi_table_property.details, details_df[['Table', 'Metric', 'Score']]
        )

    def test_get_score_converts_nan_to_none(self):
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

        base_multi_table_property = BaseMultiTableProperty()
        base_multi_table_property._generate_details = Mock()
        base_multi_table_property._compute_average = Mock(return_value=0.7)

        details_df = pd.DataFrame({
            'Table': ['Table1', 'Table2', 'Table3'],
            'Metric': ['Metric1', 'Metric2', 'Metric3'],
            'Score': [1.0, 1.0, 1.0],
            'Error': [None, np.nan, 'Some Error']
        })
        base_multi_table_property.details = details_df.copy()

        # Run
        result = base_multi_table_property.get_score(
            real_data, synthetic_data, metadata, progress_bar
        )

        # Assert
        expected_details = pd.DataFrame({
            'Table': ['Table1', 'Table2', 'Table3'],
            'Metric': ['Metric1', 'Metric2', 'Metric3'],
            'Score': [1.0, 1.0, 1.0],
            'Error': [None, None, 'Some Error']
        })
        base_multi_table_property._generate_details.assert_called_once_with(
            real_data, synthetic_data, metadata, progress_bar
        )
        base_multi_table_property._compute_average.assert_called_once()
        assert result == 0.7
        assert base_multi_table_property.is_computed is True
        pd.testing.assert_frame_equal(base_multi_table_property.details, expected_details)

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

    def test_get_details(self):
        """Test the ``get_details`` method."""
        # Setup
        details = pd.DataFrame({
            'Table': ['table1', 'table2', 'table3'],
            'Column 1': ['col1', 'col2', 'col3'],
            'Column 2': ['colA', 'colB', 'colC'],
            'Score': [0, 0.5, 1.0],
            'Error': [None, None, None]
        })

        base_property = BaseMultiTableProperty()
        base_property.details = details

        # Run
        full_details = base_property.get_details()
        table_details = base_property.get_details('table2')

        # Assert
        expected_table_details = pd.DataFrame({
            'Table': ['table2'],
            'Column 1': ['col2'],
            'Column 2': ['colB'],
            'Score': [0.5],
            'Error': [None]
        }, index=[1])
        pd.testing.assert_frame_equal(details, full_details)
        pd.testing.assert_frame_equal(table_details, expected_table_details)

    def test_get_details_with_parent_child(self):
        """Test ``get_details`` with properties with parent/child relationships."""
        # Setup
        details = pd.DataFrame({
            'Parent Table': ['table1', 'table3', 'table3'],
            'Child Table': ['table2', 'table2', 'table4'],
            'Column 1': ['col1', 'col2', 'col3'],
            'Column 2': ['colA', 'colB', 'colC'],
            'Score': [0, 0.5, 1.0],
            'Error': [None, None, None]
        })

        base_property = BaseMultiTableProperty()
        base_property.details = details

        # Run
        full_details = []
        table_details = []
        for prop in ['relationship', 'inter_table_column_pair']:
            base_property._num_iteration_case = prop
            full_details.append(base_property.get_details())
            table_details.append(base_property.get_details('table2'))

        # Assert
        expected_table_details = pd.DataFrame({
            'Parent Table': ['table1', 'table3'],
            'Child Table': ['table2', 'table2'],
            'Column 1': ['col1', 'col2'],
            'Column 2': ['colA', 'colB'],
            'Score': [0.0, 0.5],
            'Error': [None, None]
        })
        for detail_df in full_details:
            pd.testing.assert_frame_equal(detail_df, details)

        for detail_df in table_details:
            pd.testing.assert_frame_equal(detail_df, expected_table_details)
