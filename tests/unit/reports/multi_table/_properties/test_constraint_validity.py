"""Test ConstraintValidity property."""

from copy import deepcopy
from unittest.mock import Mock, patch

import numpy as np
import pandas as pd
import pytest

from sdmetrics.errors import VisualizationUnavailableError
from sdmetrics.reports.multi_table._properties import BaseMultiTableProperty, ConstraintValidity


@pytest.fixture
def data():
    return {
        'users': pd.DataFrame({'user_id': [0, 1], 'age': [20, 30]}),
        'sessions': pd.DataFrame({
            'session_id': [0, 1, 2],
            'user_id': [0, 0, 1],
            'device': ['mobile', 'tablet', 'mobile'],
            'os': ['android', 'ios', 'android'],
        }),
    }


@pytest.fixture
def metadata():
    return {
        'tables': {
            'users': {
                'primary_key': 'user_id',
                'columns': {'user_id': {'sdtype': 'id'}, 'age': {'sdtype': 'numerical'}},
            },
            'sessions': {
                'primary_key': 'session_id',
                'columns': {
                    'session_id': {'sdtype': 'id'},
                    'user_id': {'sdtype': 'id'},
                    'device': {'sdtype': 'categorical'},
                    'os': {'sdtype': 'categorical'},
                },
            },
        },
        'relationships': [
            {
                'parent_table_name': 'users',
                'child_table_name': 'sessions',
                'parent_primary_key': 'user_id',
                'child_foreign_key': 'user_id',
            }
        ],
    }


@pytest.fixture
def constraints():
    return [
        {
            'class_name': 'FixedCombinations',
            'parameters': {'table_name': 'sessions', 'column_names': ['device', 'os']},
        },
        {
            'class_name': 'Inequality',
            'parameters': {
                'table_name': 'users',
                'low_column_name': 'user_id',
                'high_column_name': 'age',
            },
        },
    ]


class TestConstraintValidity:
    def test__init__(self):
        """Test the ``__init__`` method."""
        # Setup
        constraint_validity = ConstraintValidity()

        # Assert
        assert isinstance(constraint_validity, BaseMultiTableProperty)
        assert constraint_validity._num_iteration_case == 'constraint'
        assert constraint_validity.is_computed is False
        assert constraint_validity.details.empty

    def test__get_num_iterations(self, metadata, constraints):
        """Test ``_get_num_iterations`` returns one iteration per constraint."""
        # Setup
        constraint_validity = ConstraintValidity()

        # Run
        num_iterations = constraint_validity._get_num_iterations(metadata, constraints)
        num_iterations_empty = constraint_validity._get_num_iterations(metadata, [])
        num_iterations_none = constraint_validity._get_num_iterations(metadata)

        # Assert
        assert num_iterations == 2
        assert num_iterations_empty == 0
        assert num_iterations_none == 0

    @patch('sdmetrics.reports.multi_table._properties.constraint_validity.ConstraintAdherence')
    def test__generate_details(self, mock_constraint_adherence, data, metadata, constraints):
        """Test ``_generate_details`` computes ``ConstraintAdherence`` for every constraint."""
        # Setup
        synthetic_data = deepcopy(data)
        mock_constraint_adherence.__name__ = 'ConstraintAdherence'
        mock_constraint_adherence.compute.side_effect = [0.8, 0.4]
        constraint_validity = ConstraintValidity()
        progress_bar = Mock()
        expected_details = pd.DataFrame({
            'Constraint': ['FixedCombinations', 'Inequality'],
            'Metric': ['ConstraintAdherence', 'ConstraintAdherence'],
            'Parameters': [constraints[0]['parameters'], constraints[1]['parameters']],
            'Score': [0.8, 0.4],
            'Error': [None, None],
        })

        # Run
        constraint_validity._generate_details(
            data, synthetic_data, metadata, constraints, progress_bar
        )

        # Assert
        pd.testing.assert_frame_equal(constraint_validity.details, expected_details)
        assert mock_constraint_adherence.compute.call_count == 2
        mock_constraint_adherence.compute.assert_any_call(
            data, synthetic_data, metadata, constraints[0]
        )
        mock_constraint_adherence.compute.assert_any_call(
            data, synthetic_data, metadata, constraints[1]
        )
        assert progress_bar.update.call_count == 2

    @patch('sdmetrics.reports.multi_table._properties.constraint_validity.ConstraintAdherence')
    def test__generate_details_with_errors(
        self, mock_constraint_adherence, data, metadata, constraints
    ):
        """Test ``_generate_details`` stores the error when the metric crashes."""
        # Setup
        synthetic_data = deepcopy(data)
        mock_constraint_adherence.__name__ = 'ConstraintAdherence'
        mock_constraint_adherence.compute.side_effect = [ValueError('error 1'), 0.4]
        constraint_validity = ConstraintValidity()
        progress_bar = Mock()
        expected_details = pd.DataFrame({
            'Constraint': ['FixedCombinations', 'Inequality'],
            'Metric': ['ConstraintAdherence', 'ConstraintAdherence'],
            'Parameters': [constraints[0]['parameters'], constraints[1]['parameters']],
            'Score': [np.nan, 0.4],
            'Error': ['ValueError: error 1', None],
        })

        # Run
        constraint_validity._generate_details(
            data, synthetic_data, metadata, constraints, progress_bar
        )

        # Assert
        pd.testing.assert_frame_equal(constraint_validity.details, expected_details)
        assert progress_bar.update.call_count == 2

    @patch('sdmetrics.reports.multi_table._properties.constraint_validity.ConstraintAdherence')
    def test__generate_details_invalid_constraint(
        self, mock_constraint_adherence, data, synthetic_data, metadata
    ):
        """Test ``_generate_details`` handles constraints that are not dictionaries."""
        # Setup
        mock_constraint_adherence.__name__ = 'ConstraintAdherence'
        mock_constraint_adherence.compute.side_effect = [ValueError('not a dict')]
        constraint_validity = ConstraintValidity()

        # Run
        constraint_validity._generate_details(data, synthetic_data, metadata, ['invalid'])

        # Assert
        expected_details = pd.DataFrame({
            'Constraint': [None],
            'Metric': ['ConstraintAdherence'],
            'Parameters': [None],
            'Score': [np.nan],
            'Error': ['ValueError: not a dict'],
        })
        pd.testing.assert_frame_equal(constraint_validity.details, expected_details)

    @patch('sdmetrics.reports.multi_table._properties.constraint_validity.ConstraintAdherence')
    def test_get_score(
        self, mock_constraint_adherence, data, synthetic_data, metadata, constraints
    ):
        """Test ``get_score`` averages the constraint scores and drops the empty error column."""
        # Setup
        mock_constraint_adherence.__name__ = 'ConstraintAdherence'
        mock_constraint_adherence.compute.side_effect = [0.8, 0.4]
        constraint_validity = ConstraintValidity()
        progress_bar = Mock()

        # Run
        score = constraint_validity.get_score(
            data, synthetic_data, metadata, constraints, progress_bar
        )

        # Assert
        expected_details = pd.DataFrame({
            'Constraint': ['FixedCombinations', 'Inequality'],
            'Metric': ['ConstraintAdherence', 'ConstraintAdherence'],
            'Parameters': [constraints[0]['parameters'], constraints[1]['parameters']],
            'Score': [0.8, 0.4],
        })
        assert score == pytest.approx(0.6)
        assert constraint_validity.is_computed is True
        pd.testing.assert_frame_equal(constraint_validity.details, expected_details)
        assert progress_bar.update.call_count == 2

    @patch('sdmetrics.reports.multi_table._properties.constraint_validity.ConstraintAdherence')
    def test_get_score_with_errors(
        self, mock_constraint_adherence, data, synthetic_data, metadata, constraints
    ):
        """Test ``get_score`` keeps the error column and ignores NaN scores in the average."""
        # Setup
        mock_constraint_adherence.__name__ = 'ConstraintAdherence'
        mock_constraint_adherence.compute.side_effect = [ValueError('error 1'), 0.4]
        constraint_validity = ConstraintValidity()

        # Run
        score = constraint_validity.get_score(data, synthetic_data, metadata, constraints)

        # Assert
        expected_details = pd.DataFrame({
            'Constraint': ['FixedCombinations', 'Inequality'],
            'Metric': ['ConstraintAdherence', 'ConstraintAdherence'],
            'Parameters': [constraints[0]['parameters'], constraints[1]['parameters']],
            'Score': [np.nan, 0.4],
            'Error': ['ValueError: error 1', None],
        })
        assert score == 0.4
        pd.testing.assert_frame_equal(constraint_validity.details, expected_details)

    @pytest.mark.parametrize('constraints', [[], None])
    def test_get_score_without_constraints(self, data, synthetic_data, metadata, constraints):
        """Test ``get_score`` returns NaN and empty details when there are no constraints."""
        # Setup
        constraint_validity = ConstraintValidity()
        progress_bar = Mock()

        # Run
        score = constraint_validity.get_score(
            data, synthetic_data, metadata, constraints, progress_bar
        )

        # Assert
        assert pd.isna(score)
        assert constraint_validity.is_computed is True
        assert list(constraint_validity.details.columns) == [
            'Constraint',
            'Metric',
            'Parameters',
            'Score',
        ]
        assert constraint_validity.details.empty
        progress_bar.update.assert_not_called()

    def test_get_details(self):
        """Test ``get_details`` returns a copy of the details."""
        # Setup
        constraint_validity = ConstraintValidity()
        constraint_validity.details = pd.DataFrame({
            'Constraint': ['FixedCombinations'],
            'Metric': ['ConstraintAdherence'],
            'Parameters': [{'table_name': 'sessions', 'column_names': ['device', 'os']}],
            'Score': [1.0],
        })

        # Run
        details = constraint_validity.get_details()

        # Assert
        pd.testing.assert_frame_equal(details, constraint_validity.details)
        assert details is not constraint_validity.details

    def test_get_details_with_table_name(self):
        """Test ``get_details`` raises an error when a table name is given."""
        # Setup
        constraint_validity = ConstraintValidity()
        expected_message = (
            'The Constraint Validity property does not break down its details by table. '
            "Please call 'get_details' without a table name."
        )

        # Run and Assert
        with pytest.raises(ValueError, match=expected_message):
            constraint_validity.get_details('users')

    @pytest.mark.parametrize('table_name', [None, 'users'])
    def test_get_visualization(self, table_name):
        """Test ``get_visualization`` raises a friendly error."""
        # Setup
        constraint_validity = ConstraintValidity()
        expected_message = (
            'Error: No visualization is available for Constraint Validity. To see the '
            "detailed score breakdowns, use the 'get_details' function."
        )

        # Run and Assert
        with pytest.raises(VisualizationUnavailableError, match=expected_message):
            constraint_validity.get_visualization(table_name)
