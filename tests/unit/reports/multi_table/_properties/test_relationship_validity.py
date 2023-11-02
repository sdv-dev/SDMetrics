"""Test multi-table RelationshipValidity properties."""

from unittest.mock import Mock, patch

import numpy as np
import pandas as pd
import pytest
from plotly.graph_objects import Figure

from sdmetrics.reports.multi_table._properties.relationship_validity import RelationshipValidity


@pytest.fixture()
def real_data_fixture():
    real_user_df = pd.DataFrame({
        'user_id': ['user1', 'user2'],
        'columnA': ['A', 'B'],
        'columnB': [np.nan, 1.0]
    })
    real_session_df = pd.DataFrame({
        'session_id': ['session1', 'session2', 'session3'],
        'user_id': ['user1', 'user1', 'user2'],
        'columnC': ['X', 'Y', 'Z'],
        'columnD': [4.0, 6.0, 7.0]
    })
    return {'users': real_user_df, 'sessions': real_session_df}


@pytest.fixture()
def synthetic_data_fixture():
    synthetic_user_df = pd.DataFrame({
        'user_id': ['user1', 'user2'],
        'columnA': ['A', 'A'],
        'columnB': [0.5, np.nan]
    })
    synthetic_session_df = pd.DataFrame({
        'session_id': ['session1', 'session2', 'session3'],
        'user_id': ['user1', 'user1', 'user2'],
        'columnC': ['X', 'Z', 'Y'],
        'columnD': [3.6, 5.0, 6.0]
    })
    return {'users': synthetic_user_df, 'sessions': synthetic_session_df}


@pytest.fixture()
def metadata_fixture():
    return {
        'tables': {
            'users': {
                'primary_key': 'user_id',
                'columns': {
                    'user_id': {'sdtype': 'id'},
                    'columnA': {'sdtype': 'categorical'},
                    'columnB': {'sdtype': 'numerical'}
                },
            },
            'sessions': {
                'primary_key': 'session_id',
                'columns': {
                    'session_id': {'sdtype': 'id'},
                    'user_id': {'sdtype': 'id'},
                    'columnC': {'sdtype': 'categorical'},
                    'columnD': {'sdtype': 'numerical'}
                }
            }
        },
        'relationships': [
            {
                'parent_table_name': 'users',
                'child_table_name': 'sessions',
                'parent_primary_key': 'user_id',
                'child_foreign_key': 'user_id'
            }
        ]
    }


class TestRelationshipValidity:

    def test__extract_tuple(self, real_data_fixture):
        """Test the ``_extract_tuple`` method."""
        # Setup
        relationship_validity = RelationshipValidity()
        real_data = real_data_fixture
        relation = {
            'parent_table_name': 'users',
            'child_table_name': 'sessions',
            'parent_primary_key': 'user_id',
            'child_foreign_key': 'user_id'
        }

        # Run
        real_columns = relationship_validity._extract_tuple(real_data, relation)

        # Assert
        assert real_columns == (real_data['users']['user_id'], real_data['sessions']['user_id'])

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
        relationship_validity = RelationshipValidity()

        # Run
        num_iterations = relationship_validity._get_num_iterations(metadata)

        # Assert
        assert num_iterations == 3

    @patch('sdmetrics.reports.multi_table._properties.relationship_validity.'
           'CardinalityBoundaryAdherence')
    @patch('sdmetrics.reports.multi_table._properties.relationship_validity.ReferentialIntegrity')
    def test_get_score(
        self, mock_referentialintegrity, mock_cardinalityboundaryadherence,
        real_data_fixture, synthetic_data_fixture, metadata_fixture
    ):
        """Test the ``get_score`` function.

        Test that when given a ``progress_bar`` and relationships, this calls
        ``CardinalityBoundaryAdherence`` and ``ReferentialIntegrity`` compute
        method for each relationship.
        """
        # Setup
        mock_referentialintegrity.compute.return_value = 0.7
        mock_referentialintegrity.__name__ = 'ReferentialIntegrity'
        mock_cardinalityboundaryadherence.compute.return_value = 0.3
        mock_cardinalityboundaryadherence.__name__ = 'CardinalityBoundaryAdherence'
        mock_compute_average = Mock(return_value=0.5)
        relationship_validity = RelationshipValidity()
        relationship_validity._compute_average = mock_compute_average
        progress_bar = Mock()

        real_data = real_data_fixture
        synthetic_data = synthetic_data_fixture
        metadata = metadata_fixture

        # Run
        score = relationship_validity.get_score(
            real_data=real_data, synthetic_data=synthetic_data, metadata=metadata,
            progress_bar=progress_bar
        )

        # Assert
        expected_details_property = pd.DataFrame({
            'Parent Table': ['users', 'users'],
            'Child Table': ['sessions', 'sessions'],
            'Primary Key': ['user_id', 'user_id'],
            'Foreign Key': ['user_id', 'user_id'],
            'Metric': ['ReferentialIntegrity', 'CardinalityBoundaryAdherence'],
            'Score': [0.7, 0.3],
        })

        assert score == 0.5
        progress_bar.update.assert_called()
        progress_bar.update.assert_called_once()
        mock_compute_average.assert_called_once()
        pd.testing.assert_frame_equal(relationship_validity.details, expected_details_property)

    @patch('sdmetrics.reports.multi_table._properties.relationship_validity.'
           'CardinalityBoundaryAdherence')
    @patch('sdmetrics.reports.multi_table._properties.relationship_validity.ReferentialIntegrity')
    def test_get_score_raises_errors(
        self, mock_referentialintegrity, mock_cardinalityboundaryadherence,
        real_data_fixture, synthetic_data_fixture, metadata_fixture
    ):
        """Test the ``get_score`` when ``ReferentialIntegrity`` or
        ``CardinalityBoundaryAdherence`` crashes"""
        # Setup
        mock_referentialintegrity.compute.side_effect = [ValueError('error 1')]
        mock_referentialintegrity.__name__ = 'ReferentialIntegrity'
        mock_cardinalityboundaryadherence.compute.side_effect = [ValueError('error 2')]
        mock_cardinalityboundaryadherence.__name__ = 'CardinalityBoundaryAdherence'
        relationship_validity = RelationshipValidity()
        progress_bar = Mock()

        real_data = real_data_fixture
        synthetic_data = synthetic_data_fixture
        metadata = metadata_fixture

        # Run
        score = relationship_validity.get_score(
            real_data=real_data, synthetic_data=synthetic_data, metadata=metadata,
            progress_bar=progress_bar
        )

        # Assert
        expected_details_property = pd.DataFrame({
            'Parent Table': ['users', 'users'],
            'Child Table': ['sessions', 'sessions'],
            'Primary Key': ['user_id', 'user_id'],
            'Foreign Key': ['user_id', 'user_id'],
            'Metric': ['ReferentialIntegrity', 'CardinalityBoundaryAdherence'],
            'Score': [np.nan, np.nan],
            'Error': ['ValueError: error 1', 'ValueError: error 2']
        })

        assert pd.isna(score)
        pd.testing.assert_frame_equal(relationship_validity.details, expected_details_property)
        progress_bar.update.assert_called()
        progress_bar.update.assert_called_once()

    def test_get_details_with_table_name(self):
        """Test the ``get_details`` method.

        Test that the method returns the correct details for the given table name,
        either from the child or parent table.
        """
        # Setup
        relationship_validity = RelationshipValidity()
        relationship_validity.details = pd.DataFrame({
            'Child Table': ['users_child', 'sessions_child'],
            'Parent Table': ['users_parent', 'sessions_parent'],
            'Primary Key': ['user_id', 'user_id'],
            'Foreign Key': ['user_id', 'user_id'],
            'Metric': ['ReferentialIntegrity', 'CardinalityBoundaryAdherence'],
            'Score': [1.0, 0.5],
            'Error': [None, 'Some error']
        })

        # Run
        details_users_child = relationship_validity.get_details('users_child')
        details_sessions_parent = relationship_validity.get_details('sessions_parent')

        # Assert for child table
        assert details_users_child.equals(pd.DataFrame({
            'Child Table': ['users_child'],
            'Parent Table': ['users_parent'],
            'Primary Key': ['user_id'],
            'Foreign Key': ['user_id'],
            'Metric': ['ReferentialIntegrity'],
            'Score': [1.0],
            'Error': [None]
        }, index=[0]))

        # Assert for parent table
        assert details_sessions_parent.equals(pd.DataFrame({
            'Child Table': ['sessions_child'],
            'Parent Table': ['sessions_parent'],
            'Primary Key': ['user_id'],
            'Foreign Key': ['user_id'],
            'Metric': ['CardinalityBoundaryAdherence'],
            'Score': [0.5],
            'Error': ['Some error']
        }, index=[1]))

    def test_get_table_relationships_plot(self):
        """Test the ``_get_table_relationships_plot`` method.

        Test that the method returns the correct plotly figure for the given table name.
        """
        # Setup
        instance = RelationshipValidity()
        instance.details = pd.DataFrame({
            'Child Table': ['users_child', 'sessions_child'],
            'Parent Table': ['users_parent', 'sessions_parent'],
            'Primary Key': ['user_id', 'user_id'],
            'Foreign Key': ['user_id', 'user_id'],
            'Metric': ['ReferentialIntegrity', 'CardinalityBoundaryAdherence'],
            'Score': [1.0, 0.5],
            'Error': [None, 'Some error']
        })

        # Run
        fig = instance._get_table_relationships_plot('users_child')

        # Assert
        assert isinstance(fig, Figure)

        expected_x = ['users_child (user_id) → users_parent']
        expected_y = [1.0]
        expected_title = 'Data Diagnostic: Relationship Validity (Average Score=1.0)'

        assert fig.data[0].x.tolist() == expected_x
        assert fig.data[0].y.tolist() == expected_y
        assert fig.layout.title.text == expected_title

    def test_get_visualization(self):
        """Test the ``get_visualization`` method."""
        # Setup
        mock__get_table_relationships_plot = Mock(side_effect=[Figure()])
        relationship_validity = RelationshipValidity()
        relationship_validity._get_table_relationships_plot = mock__get_table_relationships_plot

        # Run
        fig = relationship_validity.get_visualization('table_name')

        # Assert
        assert isinstance(fig, Figure)
        mock__get_table_relationships_plot.assert_called_once_with('table_name')
