import re
from copy import deepcopy

import numpy as np
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

    def test__validate_data_missing_table(self, data, metadata, constraint):
        """Test ``_validate_data`` errors if a reference table is not in the data."""
        # Setup
        del data['grandparent']
        expected_error = re.escape("The table 'grandparent' is missing from the data.")

        # Run and Assert
        with pytest.raises(ConstraintNotApplicableError, match=expected_error):
            constraint._validate_data(data, metadata)

    def test__validate_data_different_columns(self, data, metadata, constraint):
        """Test ``_validate_data`` errors if a reference table changed its columns."""
        # Setup
        constraint.fit(data, metadata)
        del data['grandparent']['col']
        expected_error = re.escape(
            "The columns of the table 'grandparent' do not match the ones of the real data."
        )

        # Run and Assert
        with pytest.raises(ConstraintNotApplicableError, match=expected_error):
            constraint._validate_data(data, metadata)

    def test__fit(self, data, metadata, constraint):
        """Test ``_fit`` learns the rows that the reference table holds."""
        # Run
        constraint._fit(data, metadata)

        # Assert
        assert constraint._reference_columns == {'grandparent': ['pk', 'col']}
        assert constraint._reference_rows == {
            'grandparent': {(0, 'A'), (1, 'B'), (2, 'C'), (3, 'D'), (4, 'E')}
        }

    def test__is_valid(self, data, metadata, constraint):
        """Test that every row of an unchanged reference table is valid."""
        # Setup
        constraint.fit(data, metadata)

        # Run
        valid_rows = constraint._is_valid(data, metadata)

        # Assert
        for column in valid_rows.values():
            assert all(column)

    def test__is_valid_not_fitted(self, data, metadata, constraint):
        """Test ``_is_valid`` errors if the constraint was not fitted first."""
        # Setup
        expected_error = re.escape('ReferenceTable constraint must be called with ``fit`` first.')

        # Run and Assert
        with pytest.raises(ConstraintNotApplicableError, match=expected_error):
            constraint._is_valid(data, metadata)

    def test__is_valid_with_a_changed_row(self, data, metadata, constraint):
        """Test ``_is_valid`` flags a reference row that the real data does not hold."""
        # Setup
        constraint.fit(data, metadata)
        synthetic_data = deepcopy(data)
        synthetic_data['grandparent'].loc[1, 'col'] = 'Z'

        # Run
        valid_rows = constraint._is_valid(synthetic_data, metadata)

        # Assert
        expected = pd.Series([True, False, True, True, True])
        pd.testing.assert_series_equal(valid_rows['grandparent'], expected)
        assert all(valid_rows['parent'])
        assert all(valid_rows['child'])

    def test__is_valid_with_a_new_row(self, data, metadata, constraint):
        """Test ``_is_valid`` flags a reference row that is not in the real data."""
        # Setup
        constraint.fit(data, metadata)
        synthetic_data = deepcopy(data)
        synthetic_data['grandparent'] = pd.DataFrame({
            'pk': [0, 1, 9],
            'col': ['A', 'B', 'Z'],
        })

        # Run
        valid_rows = constraint._is_valid(synthetic_data, metadata)

        # Assert
        pd.testing.assert_series_equal(valid_rows['grandparent'], pd.Series([True, True, False]))

    def test__is_valid_ignores_the_row_order(self, data, metadata, constraint):
        """Test ``_is_valid`` does not care about the order of the reference rows."""
        # Setup
        constraint.fit(data, metadata)
        synthetic_data = deepcopy(data)
        synthetic_data['grandparent'] = (
            synthetic_data['grandparent'].iloc[::-1].reset_index(drop=True)
        )

        # Run
        valid_rows = constraint._is_valid(synthetic_data, metadata)

        # Assert
        assert all(valid_rows['grandparent'])

    def test__is_valid_with_missing_values(self, data, metadata, constraint):
        """Test ``_is_valid`` matches two reference rows that are null in the same column."""
        # Setup
        data['grandparent']['col'] = ['A', None, 'C', 'D', 'E']
        constraint.fit(data, metadata)
        synthetic_data = deepcopy(data)
        synthetic_data['grandparent']['col'] = ['A', np.nan, 'C', 'D', 'E']

        # Run
        valid_rows = constraint._is_valid(synthetic_data, metadata)

        # Assert
        assert all(valid_rows['grandparent'])

    def test_get_score(self, data, metadata, constraint):
        """Test ``get_score`` returns the proportion of valid rows."""
        # Setup
        constraint.fit(data, metadata)

        # Run and Assert
        assert constraint.get_score(data, metadata) == 1.0

    def test_get_score_with_a_changed_row(self, data, metadata, constraint):
        """Test ``get_score`` only counts the rows of the reference tables."""
        # Setup
        constraint.fit(data, metadata)
        synthetic_data = deepcopy(data)
        synthetic_data['grandparent'].loc[1, 'col'] = 'Z'

        # Run and Assert
        assert constraint.get_score(synthetic_data, metadata) == 4 / 5

    def test_get_score_empty_tables(self, data, metadata, constraint):
        """Test ``get_score`` returns NaN when there are no rows to check."""
        # Setup
        constraint.fit(data, metadata)
        data = {table: table_data.iloc[:0] for table, table_data in data.items()}

        # Run and Assert
        assert pd.isna(constraint.get_score(data, metadata))
