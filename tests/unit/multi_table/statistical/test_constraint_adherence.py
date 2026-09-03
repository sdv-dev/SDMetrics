import re

import pandas as pd
import pytest

from sdmetrics.multi_table import ConstraintAdherence


@pytest.fixture
def real_data():
    return {
        'table': pd.DataFrame({
            'a': ['a', 'b', 'c'],
            'b': ['d', 'e', 'f'],
            'c': ['g', 'h', 'i'],
        })
    }


@pytest.fixture
def metadata():
    return {
        'tables': {
            'table': {
                'columns': {
                    'a': {'sdtype': 'categorical'},
                    'b': {'sdtype': 'categorical'},
                    'c': {'sdtype': 'categorical'},
                }
            }
        }
    }


@pytest.fixture
def constraint():
    return {
        'class_name': 'FixedCombinations',
        'parameters': {
            'table_name': 'table',
            'column_names': ['a', 'b'],
        },
    }


class TestConstraintAdherence:
    def test_compute(self, real_data, metadata, constraint):
        """Test ``compute`` returns the proportion of valid synthetic rows."""
        # Setup
        synthetic_data = {
            'table': pd.DataFrame({
                'a': ['a', 'b', 'c'],
                'b': ['d', 'f', 'e'],
                'c': ['g', 'h', 'i'],
            })
        }

        # Run
        score = ConstraintAdherence.compute(real_data, synthetic_data, metadata, constraint)

        # Assert
        assert score == 1 / 3

    def test_compute_ignores_unconstrained_columns(self, real_data, metadata, constraint):
        """Test ``compute`` only looks at the columns named by the constraint."""
        # Setup
        synthetic_data = {'table': real_data['table'].copy()}
        synthetic_data['table']['c'] = ['z', 'z', 'z']

        # Run
        score = ConstraintAdherence.compute(real_data, synthetic_data, metadata, constraint)

        # Assert
        assert score == 1.0

    def test_compute_invalid_real_data(self):
        """Test ``compute`` warns about the real data but still scores the synthetic data."""
        # Setup
        real_data = {'table': pd.DataFrame({'low': [1, 2, 3], 'high': [2, 1, 4]})}
        synthetic_data = {'table': pd.DataFrame({'low': [1, 2, 3], 'high': [2, 3, 4]})}
        metadata = {
            'tables': {
                'table': {
                    'columns': {
                        'low': {'sdtype': 'numerical'},
                        'high': {'sdtype': 'numerical'},
                    }
                }
            }
        }
        constraint = {
            'class_name': 'Inequality',
            'parameters': {
                'table_name': 'table',
                'low_column_name': 'low',
                'high_column_name': 'high',
            },
        }

        # Run
        warning_message = 'The real data does not adhere'
        with pytest.warns(UserWarning, match=warning_message):
            score = ConstraintAdherence.compute(real_data, synthetic_data, metadata, constraint)

        # Assert
        assert score == 1.0

    def test_compute_missing_table(self, real_data, metadata, constraint):
        """Test ``compute`` warns and returns NaN if the constraint can't be checked."""
        # Setup
        constraint['parameters']['table_name'] = 'MissingTable'
        expected_msg = (
            'Unable to check the constraint against the real data: '
            "The table 'MissingTable' is missing from the data."
        )

        # Run
        with pytest.warns(UserWarning, match=expected_msg):
            score = ConstraintAdherence.compute(real_data, real_data, metadata, constraint)

        # Assert
        assert pd.isna(score)

    def test_compute_missing_column(self, real_data, metadata, constraint):
        """Test ``compute`` warns and returns NaN if a constrained column is missing."""
        # Setup
        constraint['parameters']['column_names'] = ['a', 'missing']
        expected_msg = re.escape(
            'Unable to check the constraint against the real data: '
            "The column(s) 'missing' are missing from the table 'table'."
        )

        # Run
        with pytest.warns(UserWarning, match=expected_msg):
            score = ConstraintAdherence.compute(real_data, real_data, metadata, constraint)

        # Assert
        assert pd.isna(score)

    def test_compute_unsupported_constraint(self, real_data, metadata):
        """Test ``compute`` warns and returns NaN if the constraint is not supported."""
        # Setup
        constraint = {'class_name': 'Unsupported', 'parameters': {}}

        # Run
        with pytest.warns(UserWarning, match='Unable to check the constraint'):
            score = ConstraintAdherence.compute(real_data, real_data, metadata, constraint)

        # Assert
        assert pd.isna(score)
