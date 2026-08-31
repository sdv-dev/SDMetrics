import re

import pandas as pd
import pytest

from sdmetrics.multi_table.statistical.constraints import ReferenceTable
from sdmetrics.multi_table.statistical.constraints.error import ConstraintNotApplicableError


@pytest.fixture
def metadata():
    """Metadata for the test.

    It has the following relationships:
    - grandparent -> parent
    - parent -> child
    - grandparent -> child
    """
    return {
        'tables': {
            'grandparent': {
                'columns': {'pk': {'sdtype': 'id'}, 'col': {'sdtype': 'categorical'}},
                'primary_key': 'pk',
            },
            'parent': {
                'columns': {
                    'pk': {'sdtype': 'id'},
                    'fk': {'sdtype': 'id'},
                    'col': {'sdtype': 'categorical'},
                },
                'primary_key': 'pk',
            },
            'child': {
                'columns': {
                    'pk': {'sdtype': 'id'},
                    'fk_parent': {'sdtype': 'id'},
                    'fk_grandparent': {'sdtype': 'id'},
                    'col': {'sdtype': 'categorical'},
                },
                'primary_key': 'pk',
            },
        },
        'relationships': [
            {
                'parent_table_name': 'grandparent',
                'child_table_name': 'parent',
                'parent_primary_key': 'pk',
                'child_foreign_key': 'fk',
            },
            {
                'parent_table_name': 'parent',
                'child_table_name': 'child',
                'parent_primary_key': 'pk',
                'child_foreign_key': 'fk_parent',
            },
            {
                'parent_table_name': 'grandparent',
                'child_table_name': 'child',
                'parent_primary_key': 'pk',
                'child_foreign_key': 'fk_grandparent',
            },
        ],
    }


@pytest.fixture
def data():
    return {
        'grandparent': pd.DataFrame({'pk': range(5), 'col': ['A', 'B', 'C', 'D', 'E']}),
        'parent': pd.DataFrame({
            'pk': range(5),
            'fk': [0, 1, 1, 2, 4],
            'col': ['A', 'B', 'C', 'D', 'E'],
        }),
        'child': pd.DataFrame({
            'pk': range(5),
            'fk_parent': [0, 1, 2, 3, 4],
            'fk_grandparent': [0, 1, 1, 2, 4],
            'col': ['X', 'Y', 'Z', 'X', 'Y'],
        }),
    }


@pytest.fixture
def constraint():
    return ReferenceTable(reference_table_names=['grandparent'])


class TestReferenceTable:
    def test___init__(self, constraint):
        """Test the ``__init__`` method sets the parameters."""
        # Assert
        assert constraint.reference_table_names == ['grandparent']

    def test___init___invalid_reference_table_type(self):
        """Test the ``__init__`` method when reference_table_names is not a list."""
        # Run and Assert
        with pytest.raises(ValueError, match="'reference_table_names' must be a list of strings."):
            ReferenceTable('not_a_list')

    def test___init___invalid_reference_table_names(self):
        """Test the ``__init__`` method when reference_table_names is not a list of strings."""
        # Run and Assert
        with pytest.raises(ValueError, match="'reference_table_names' must be a list of strings."):
            ReferenceTable(['string', 10])

    def test__validate_constraint_with_metadata(self, metadata, constraint):
        """Test ``_validate_constraint_with_metadata`` passes for a table with no parent."""
        # Run and Assert
        constraint._validate_constraint_with_metadata(metadata)

    def test__validate_constraint_with_metadata_reference_parent(self, metadata):
        """Test a reference table may be the child of another reference table."""
        # Setup
        instance = ReferenceTable(reference_table_names=['grandparent', 'parent'])

        # Run and Assert
        instance._validate_constraint_with_metadata(metadata)

    def test__validate_constraint_with_metadata_every_table(self, metadata):
        """Test every table of the dataset may be a reference table."""
        # Setup
        instance = ReferenceTable(reference_table_names=['grandparent', 'parent', 'child'])

        # Run and Assert
        instance._validate_constraint_with_metadata(metadata)

    def test__validate_constraint_with_metadata_missing_table(self, metadata):
        """Test ``_validate_constraint_with_metadata`` errors if a table is not in the metadata."""
        # Setup
        instance = ReferenceTable(reference_table_names=['City', 'Country'])
        expected_error = re.escape(
            "Reference table(s) '['City', 'Country']' missing from metadata."
        )

        # Run and Assert
        with pytest.raises(ConstraintNotApplicableError, match=expected_error):
            instance._validate_constraint_with_metadata(metadata)

    def test__validate_constraint_with_metadata_non_reference_parent(self, metadata):
        """Test ``_validate_constraint_with_metadata`` errors on a non reference parent."""
        # Setup
        instance = ReferenceTable(reference_table_names=['parent'])
        expected_error = re.escape(
            'Reference tables cannot be children of non-reference tables. The following '
            "child-parent pairs are invalid: '[('parent', 'grandparent')]'"
        )

        # Run and Assert
        with pytest.raises(ConstraintNotApplicableError, match=expected_error):
            instance._validate_constraint_with_metadata(metadata)

    def test__validate_constraint_with_metadata_several_non_reference_parents(self, metadata):
        """Test ``_validate_constraint_with_metadata`` reports every invalid pair."""
        # Setup
        instance = ReferenceTable(reference_table_names=['child'])
        expected_error = re.escape(
            'Reference tables cannot be children of non-reference tables. The following '
            "child-parent pairs are invalid: '[('child', 'grandparent'), ('child', 'parent')]'"
        )

        # Run and Assert
        with pytest.raises(ConstraintNotApplicableError, match=expected_error):
            instance._validate_constraint_with_metadata(metadata)

    def test__is_valid(self, data, constraint):
        """Test that all rows are valid."""
        # Run
        valid_rows = constraint._is_valid(data)

        # Assert
        for column in valid_rows.values():
            assert all(column)

    def test_get_score(self, data, metadata, constraint):
        """Test ``get_score`` returns the proportion of valid rows."""
        # Run and Assert
        assert constraint.get_score(data, metadata) == 1.0

    def test_get_score_empty_tables(self, data, metadata, constraint):
        """Test ``get_score`` returns NaN when there are no rows to check."""
        # Setup
        data = {table: table_data.iloc[:0] for table, table_data in data.items()}

        # Run and Assert
        assert pd.isna(constraint.get_score(data, metadata))
