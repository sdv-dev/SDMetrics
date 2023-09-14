"""Test multi-table cardinality properties."""

from unittest.mock import Mock, patch

import numpy as np
import pandas as pd
from plotly.graph_objects import Figure

from sdmetrics.reports.multi_table._properties.cardinality import Cardinality


class TestCardinality:

    def test__get_num_iteration(self):
        """Test the ``_get_num_iterations`` method."""
        # Setup
        metadata = {
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
        cardinality = Cardinality()

        # Run
        num_iterations = cardinality._get_num_iterations(metadata)

        # Assert
        assert num_iterations == 3

    @patch('sdmetrics.reports.multi_table._properties.cardinality.CardinalityShapeSimilarity')
    def test_get_score(self, mock_cardinalityshapesimilarity):
        """Test the ``get_score`` function.

        Test that when given a ``progress_bar`` and relationships, this calls
        ``CardinalityShapeSimilarity`` compute breakdown for each relationship.
        """
        # Setup
        mock_cardinalityshapesimilarity.compute.side_effect = [1, 0.25]
        mock_compute_average = Mock(return_value=0.625)
        cardinality = Cardinality()
        cardinality._compute_average = mock_compute_average
        progress_bar = Mock()
        relationships = [
            {'child_table_name': 'users_child', 'parent_table_name': 'users_parent'},
            {'child_table_name': 'sessions_child', 'parent_table_name': 'sessions_parent'}
        ]
        metadata = {'relationships': relationships}

        # Run
        score = cardinality.get_score('real_data', 'synthetic_data', metadata, progress_bar)

        # Assert
        assert score == 0.625
        progress_bar.update.assert_called()
        assert progress_bar.update.call_count == 2
        mock_compute_average.assert_called_once_with()

    @patch('sdmetrics.reports.multi_table._properties.cardinality.CardinalityShapeSimilarity')
    def test_get_score_raises_errors(self, mock_cardinalityshapesimilarity):
        """Test the ``get_score`` function when CardinalityShapeSimilarity can't compute score."""
        # Setup
        mock_cardinalityshapesimilarity.compute.side_effect = [
            ValueError('Users error'),
            ValueError('Sessions error')
        ]
        cardinality = Cardinality()
        progress_bar = Mock()
        relationships = [
            {'child_table_name': 'users_child', 'parent_table_name': 'users_parent'},
            {'child_table_name': 'sessions_child', 'parent_table_name': 'sessions_parent'}
        ]
        metadata = {'relationships': relationships}

        # Run
        score = cardinality.get_score('real_data', 'synthetic_data', metadata, progress_bar)

        # Assert
        expected_details_property = pd.DataFrame({
            'Child Table': ['users_child', 'sessions_child'],
            'Parent Table': ['users_parent', 'sessions_parent'],
            'Metric': ['CardinalityShapeSimilarity', 'CardinalityShapeSimilarity'],
            'Score': [np.nan, np.nan],
            'Error': ['ValueError: Users error', 'ValueError: Sessions error']
        })

        assert pd.isna(score)
        pd.testing.assert_frame_equal(cardinality.details, expected_details_property)
        progress_bar.update.assert_called()
        assert progress_bar.update.call_count == 2

    def test_get_details_for_table_name(self):
        """Test the ``_get_details_for_table_name`` method.

        Test that the method returns the correct details for the given table name,
        either from the child or parent table.
        """
        # Setup
        cardinality = Cardinality()
        cardinality.details = pd.DataFrame({
            'Child Table': ['users_child', 'sessions_child'],
            'Parent Table': ['users_parent', 'sessions_parent'],
            'Metric': ['CardinalityShapeSimilarity', 'SomeOtherMetric'],
            'Score': [1.0, 0.5],
            'Error': [None, 'Some error']
        })

        # Run
        details_users_child = cardinality._get_details_for_table_name('users_child')
        details_sessions_parent = cardinality._get_details_for_table_name('sessions_parent')

        # Assert for child table
        assert details_users_child.equals(pd.DataFrame({
            'Child Table': ['users_child'],
            'Parent Table': ['users_parent'],
            'Metric': ['CardinalityShapeSimilarity'],
            'Score': [1.0],
            'Error': [None]
        }, index=[0]))

        # Assert for parent table
        assert details_sessions_parent.equals(pd.DataFrame({
            'Child Table': ['sessions_child'],
            'Parent Table': ['sessions_parent'],
            'Metric': ['SomeOtherMetric'],
            'Score': [0.5],
            'Error': ['Some error']
        }, index=[1]))

    def test_get_details(self):
        """Test the ``get_details`` method.

        Test that the method returns the correct details for the given property and table name.
        """
        # Setup
        mock__get_details_for_table_name = Mock(return_value='Details for table name')
        cardinality = Cardinality()
        cardinality.details = pd.DataFrame({'a': ['b']})
        cardinality._get_details_for_table_name = mock__get_details_for_table_name

        # Run
        details = cardinality.get_details('table_name')
        entire_details = cardinality.get_details()

        # Assert
        assert details == 'Details for table name'
        pd.testing.assert_frame_equal(entire_details, pd.DataFrame({'a': ['b']}))
        mock__get_details_for_table_name.assert_called_once_with('table_name')

    def test_get_table_relationships_plot(self):
        """Test the ``_get_table_relationships_plot`` method.

        Test that the method returns the correct plotly figure for the given table name.
        """
        # Setup
        instance = Cardinality()
        instance.details = pd.DataFrame({
            'Child Table': ['users_child', 'sessions_child'],
            'Parent Table': ['users_parent', 'sessions_parent'],
            'Metric': ['CardinalityShapeSimilarity', 'SomeOtherMetric'],
            'Score': [1.0, 0.5],
            'Error': [None, 'Some error']
        })

        # Run
        fig = instance._get_table_relationships_plot('users_child')

        # Assert
        assert isinstance(fig, Figure)

        expected_x = ['users_child → users_parent']
        expected_y = [1.0]
        expected_title = 'Table Relationships (Average Score=1.0)'

        assert fig.data[0].x.tolist() == expected_x
        assert fig.data[0].y.tolist() == expected_y
        assert fig.layout.title.text == expected_title

    def test_get_visualization(self):
        """Test the ``get_visualization`` method."""
        # Setup
        mock__get_table_relationships_plot = Mock(return_value='Table relationships plot')
        cardinality = Cardinality()
        cardinality._get_table_relationships_plot = mock__get_table_relationships_plot

        # Run
        fig = cardinality.get_visualization('table_name')

        # Assert
        assert fig == 'Table relationships plot'
