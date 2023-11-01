"""Test InterTableTrends multi-table class."""
import itertools
from unittest.mock import patch

import numpy as np
import pandas as pd
import pytest

from sdmetrics.reports.multi_table._properties import InterTableTrends
from tests.utils import DataFrameMatcher, IteratorMatcher, SeriesMatcher


def test__init__():
    """Test the ``__init__`` method."""
    # Setup
    column_pair_trends = InterTableTrends()

    # Assert
    assert column_pair_trends._properties == {}
    assert column_pair_trends._num_iteration_case == 'inter_table_column_pair'


@patch('sdmetrics.reports.multi_table._properties.inter_table_trends.SingleTableColumnPairTrends')
def test__generate_details(column_pair_trends_mock):
    """Test the ``get_score`` method."""
    # Setup
    instance = InterTableTrends()
    real_user_df = pd.DataFrame({
        'user_id': ['user1', 'user2'],
        'columnA': ['A', 'B'],
        'columnB': [np.nan, 1.0]
    })
    synthetic_user_df = pd.DataFrame({
        'user_id': ['user1', 'user2'],
        'columnA': ['A', 'A'],
        'columnB': [0.5, np.nan]
    })
    real_session_df = pd.DataFrame({
        'session_id': ['session1', 'session2', 'session3'],
        'user_id': ['user1', 'user1', 'user2'],
        'columnC': ['X', 'Y', 'Z'],
        'columnD': [4.0, 6.0, 7.0]
    })
    synthetic_session_df = pd.DataFrame({
        'session_id': ['session1', 'session2', 'session3'],
        'user_id': ['user1', 'user1', 'user2'],
        'columnC': ['X', 'Z', 'Y'],
        'columnD': [3.6, 5.0, 6.0]
    })

    metadata = {
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
    instanced_mock = column_pair_trends_mock.return_value
    instanced_mock._generate_details.return_value = pd.DataFrame({
        'Column 1': ['users.columnA', 'users.columnA', 'users.columnB', 'users.columnB'],
        'Column 2': [
            'sessions.columnC', 'sessions.columnD', 'sessions.columnC', 'sessions.columnB'
        ],
        'Metric': [
            'ContingencySimilarity',
            'ContingencySimilarity',
            'ContingencySimilarity',
            'CorrelationSimilarity'
        ],
        'Score': [1.0, 1.0, 0.5, 0.5],
        'Real Correlation': [None, None, None, 0.8],
        'Synthetic Correlation': [None, None, None, 0.6],
        'Error': [None, None, None, None]
    })

    # Run
    instance._generate_details(
        real_data={'users': real_user_df, 'sessions': real_session_df},
        synthetic_data={'users': synthetic_user_df, 'sessions': synthetic_session_df},
        metadata=metadata
    )

    # Assert
    expected_denormalized_real = pd.DataFrame({
        'sessions.session_id': ['session1', 'session2', 'session3'],
        'sessions.user_id': ['user1', 'user1', 'user2'],
        'sessions.columnC': ['X', 'Y', 'Z'],
        'sessions.columnD': [4.0, 6.0, 7.0],
        'users.user_id': ['user1', 'user1', 'user2'],
        'users.columnA': ['A', 'A', 'B'],
        'users.columnB': [np.nan, np.nan, 1.0]
    })
    expected_denormalized_synthetic = pd.DataFrame({
        'sessions.session_id': ['session1', 'session2', 'session3'],
        'sessions.user_id': ['user1', 'user1', 'user2'],
        'sessions.columnC': ['X', 'Z', 'Y'],
        'sessions.columnD': [3.6, 5.0, 6.0],
        'users.user_id': ['user1', 'user1', 'user2'],
        'users.columnA': ['A', 'A', 'A'],
        'users.columnB': [0.5, 0.5, np.nan]
    })
    expected_merged_metadata = {
        'primary_key': 'sessions.session_id',
        'columns': {
            'sessions.session_id': {'sdtype': 'id'},
            'sessions.user_id': {'sdtype': 'id'},
            'sessions.columnC': {'sdtype': 'categorical'},
            'sessions.columnD': {'sdtype': 'numerical'},
            'users.user_id': {'sdtype': 'id'},
            'users.columnA': {'sdtype': 'categorical'},
            'users.columnB': {'sdtype': 'numerical'},
        },
    }
    expected_column_pairs = itertools.product(
        ['users.user_id', 'users.columnA', 'users.columnB'],
        ['sessions.session_id', 'sessions.user_id', 'sessions.columnC', 'sessions.columnD']
    )
    expected_details = pd.DataFrame({
        'Parent Table': ['users', 'users', 'users', 'users'],
        'Child Table': ['sessions', 'sessions', 'sessions', 'sessions'],
        'Foreign Key': ['user_id', 'user_id', 'user_id', 'user_id'],
        'Column 1': ['columnA', 'columnA', 'columnB', 'columnB'],
        'Column 2': ['columnC', 'columnD', 'columnC', 'columnB'],
        'Metric': [
            'ContingencySimilarity',
            'ContingencySimilarity',
            'ContingencySimilarity',
            'CorrelationSimilarity'
        ],
        'Score': [1.0, 1.0, 0.5, 0.5],
        'Real Correlation': [None, None, None, 0.8],
        'Synthetic Correlation': [None, None, None, 0.6],
        'Error': [None, None, None, None]
    })
    instanced_mock._generate_details.assert_called_once_with(
        DataFrameMatcher(expected_denormalized_real),
        DataFrameMatcher(expected_denormalized_synthetic),
        expected_merged_metadata,
        progress_bar=None,
        column_pairs=IteratorMatcher(expected_column_pairs)
    )
    pd.testing.assert_frame_equal(instance.details, expected_details)


@patch('sdmetrics.reports.multi_table._properties.inter_table_trends.SingleTableColumnPairTrends')
def test__generate_details_empty_column_generate(column_pair_trends_mock):
    """Test the ``get_score`` method."""
    # Setup
    instance = InterTableTrends()
    real_user_df = pd.DataFrame({
        'user_id': ['user1', 'user2'],
    })
    synthetic_user_df = pd.DataFrame({
        'user_id': ['user1', 'user2'],
    })
    real_session_df = pd.DataFrame({
        'session_id': ['session1', 'session2', 'session3'],
        'user_id': ['user1', 'user1', 'user2'],
    })
    synthetic_session_df = pd.DataFrame({
        'session_id': ['session1', 'session2', 'session3'],
        'user_id': ['user1', 'user1', 'user2'],
    })

    metadata = {
        'tables': {
            'users': {
                'primary_key': 'user_id',
                'columns': {
                    'user_id': {'sdtype': 'id'}
                },
            },
            'sessions': {
                'primary_key': 'session_id',
                'columns': {
                    'session_id': {'sdtype': 'id'},
                    'user_id': {'sdtype': 'id'}
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
    instanced_mock = column_pair_trends_mock.return_value
    instanced_mock._generate_details.return_value = pd.DataFrame({
        'Column 1': [],
        'Column 2': [],
        'Metric': [],
        'Score': [],
        'Real Correlation': [],
        'Synthetic Correlation': [],
        'Error': []
    })

    # Run
    instance._generate_details(
        real_data={'users': real_user_df, 'sessions': real_session_df},
        synthetic_data={'users': synthetic_user_df, 'sessions': synthetic_session_df},
        metadata=metadata
    )

    # Assert
    expected_denormalized_real = pd.DataFrame({
        'sessions.session_id': ['session1', 'session2', 'session3'],
        'sessions.user_id': ['user1', 'user1', 'user2'],
        'users.user_id': ['user1', 'user1', 'user2'],
    })
    expected_denormalized_synthetic = pd.DataFrame({
        'sessions.session_id': ['session1', 'session2', 'session3'],
        'sessions.user_id': ['user1', 'user1', 'user2'],
        'users.user_id': ['user1', 'user1', 'user2'],
    })
    expected_merged_metadata = {
        'primary_key': 'sessions.session_id',
        'columns': {
            'sessions.session_id': {'sdtype': 'id'},
            'sessions.user_id': {'sdtype': 'id'},
            'users.user_id': {'sdtype': 'id'},
        },
    }
    expected_column_pairs = itertools.product(
        ['users.user_id'],
        ['sessions.session_id', 'sessions.user_id']
    )
    expected_details = pd.DataFrame({
        'Parent Table': [],
        'Child Table': [],
        'Foreign Key': [],
        'Column 1': [],
        'Column 2': [],
        'Metric': [],
        'Score': [],
        'Real Correlation': [],
        'Synthetic Correlation': [],
        'Error': []
    }).astype({
        'Parent Table': 'object',
        'Child Table': 'object',
        'Foreign Key': 'object',
        'Column 1': 'float64',
        'Column 2': 'float64',
        'Metric': 'float64',
        'Score': 'float64',
        'Real Correlation': 'float64',
        'Synthetic Correlation': 'float64',
        'Error': 'float64'
    })

    instanced_mock._generate_details.assert_called_once_with(
        DataFrameMatcher(expected_denormalized_real),
        DataFrameMatcher(expected_denormalized_synthetic),
        expected_merged_metadata,
        progress_bar=None,
        column_pairs=IteratorMatcher(expected_column_pairs)
    )
    pd.testing.assert_frame_equal(instance.details, expected_details)


@patch('sdmetrics.reports.multi_table._properties.inter_table_trends.px')
def test_get_visualization(plotly_mock):
    """Test the ``get_visualization`` method."""
    # Setup
    instance = InterTableTrends()
    instance.details = pd.DataFrame({
        'Child Table': ['users_child', 'sessions_child'],
        'Parent Table': ['users_parent', 'sessions_parent'],
        'Foreign Key': ['child_id1', 'child_id2'],
        'Column 1': ['column_a', 'column_b'],
        'Column 2': ['column_c', 'column_d'],
        'Metric': ['CorrelationSimilarity', 'ContingencySimilarity'],
        'Score': [1.0, 0.5],
        'Real Correlation': [np.nan, 1.0],
        'Synthetic Correlation': [0.8, 0.6],
        'Error': ['Some error', None]
    })
    instance.is_computed = True

    # Run
    instance.get_visualization('users_parent')

    # Assert
    expected_plot_df = pd.DataFrame({
        'Child Table': ['users_child'],
        'Parent Table': ['users_parent'],
        'Foreign Key': ['child_id1'],
        'Column 1': ['column_a'],
        'Column 2': ['column_c'],
        'Metric': ['CorrelationSimilarity'],
        'Score': [1.0],
        'Real Correlation': ['None'],
        'Synthetic Correlation': [0.8],
        'Error': ['Some error'],
        'Columns': ['users_parent.column_a, users_child.column_c']
    })
    plotly_mock.bar.assert_called_once_with(
        DataFrameMatcher(expected_plot_df),
        x='Columns',
        y='Score',
        title='Data Quality: Intertable Trends (Average Score=1.0)',
        category_orders={'group': SeriesMatcher(expected_plot_df['Columns'])},
        color='Metric',
        color_discrete_map={
            'ContingencySimilarity': '#000036',
            'CorrelationSimilarity': '#03AFF1',
        },
        pattern_shape='Metric',
        pattern_shape_sequence=['', '/'],
        custom_data=[
            'Foreign Key',
            'Metric',
            'Score',
            'Real Correlation',
            'Synthetic Correlation'
        ]
    )


@patch('sdmetrics.reports.multi_table._properties.inter_table_trends.px')
def test_get_visualization_multiple_relationships(plotly_mock):
    """Test the ``get_visualization`` method with multiple foreign keys.

    Test that the ``get_visualization`` method appends foreign keys to the column
    names when there are multiple relationships between a parent/child pair.
    """
    # Setup
    instance = InterTableTrends()
    instance.details = pd.DataFrame({
        'Parent Table': ['users', 'users'],
        'Child Table': ['sessions', 'sessions'],
        'Foreign Key': ['child_id1', 'child_id2'],
        'Column 1': ['column_a', 'column_a'],
        'Column 2': ['column_b', 'column_b'],
        'Metric': ['CorrelationSimilarity', 'CorrelationSimilarity'],
        'Score': [1.0, 0.5],
        'Real Correlation': [np.nan, 1.0],
        'Synthetic Correlation': [0.8, 0.6],
        'Error': ['Some error', None]
    })
    instance.is_computed = True

    # Run
    instance.get_visualization('sessions')

    # Assert
    expected_plot_df = pd.DataFrame({
        'Parent Table': ['users', 'users'],
        'Child Table': ['sessions', 'sessions'],
        'Foreign Key': ['child_id1', 'child_id2'],
        'Column 1': ['column_a', 'column_a'],
        'Column 2': ['column_b', 'column_b'],
        'Metric': ['CorrelationSimilarity', 'CorrelationSimilarity'],
        'Score': [1.0, 0.5],
        'Real Correlation': ['None', 1.0],
        'Synthetic Correlation': [0.8, 0.6],
        'Error': ['Some error', None],
        'Columns': [
            'users.column_a, sessions.column_b (child_id1)',
            'users.column_a, sessions.column_b (child_id2)'
        ]
    })
    plotly_mock.bar.assert_called_once_with(
        DataFrameMatcher(expected_plot_df),
        x='Columns',
        y='Score',
        title='Data Quality: Intertable Trends (Average Score=0.75)',
        category_orders={'group': SeriesMatcher(expected_plot_df['Columns'])},
        color='Metric',
        color_discrete_map={
            'ContingencySimilarity': '#000036',
            'CorrelationSimilarity': '#03AFF1',
        },
        pattern_shape='Metric',
        pattern_shape_sequence=['', '/'],
        custom_data=[
            'Foreign Key',
            'Metric',
            'Score',
            'Real Correlation',
            'Synthetic Correlation'
        ]
    )


@patch('sdmetrics.reports.multi_table._properties.inter_table_trends.px')
def test_get_visualization_error_if_not_computed(plotly_mock):
    """Test the ``get_visualization`` method."""
    # Setup
    instance = InterTableTrends()

    # Run and Assert
    error_msg = (
        'The property must be computed before getting a visualization.'
        'Please call the ``get_score`` method first.'
    )
    with pytest.raises(ValueError, match=error_msg):
        instance.get_visualization()
