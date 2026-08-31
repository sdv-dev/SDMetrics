import re

import pandas as pd
import pytest

from sdmetrics.multi_table.statistical.constraints import ForeignToPrimaryKeySubset
from sdmetrics.multi_table.statistical.constraints.error import ConstraintNotApplicableError


@pytest.fixture
def metadata():
    return {
        'tables': {
            'users': {
                'primary_key': 'user_id',
                'columns': {
                    'user_id': {'sdtype': 'id'},
                    'company_name': {'sdtype': 'company', 'pii': True},
                    'user_name': {'sdtype': 'name', 'pii': True},
                },
            },
            'transactions': {
                'primary_key': 'transaction_id',
                'columns': {
                    'transaction_id': {'sdtype': 'id'},
                    'user_id': {'sdtype': 'id'},
                    'amount': {'sdtype': 'unknown', 'pii': True},
                },
            },
        }
    }


@pytest.fixture
def data():
    users_data = pd.DataFrame({
        'user_id': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
        'company_name': [
            'TechCorp',
            'MobileInc',
            'TechCorp',
            'GadgetWorks',
            'ScreenMasters',
            'KeySolutions',
            'MouseMakers',
            'TechCorp',
            'MobileInc',
            'GadgetWorks',
        ],
        'user_name': [
            'Alice',
            'Bob',
            'Charlie',
            'David',
            'Eve',
            'Frank',
            'Grace',
            'Hank',
            'Ivy',
            'Jack',
        ],
    })

    transactions_data = pd.DataFrame({
        'transaction_id': [1001, 1002, 1003, 1004, 1005, 1006, 1007, 1008, 1009, 1010],
        'user_id': [1, 3, 8, 1, 3, 8, 1, 3, 8, 1],
        'amount': [50.00, 75.00, 20.00, 60.00, 90.00, 100.00, 150.00, 55.00, 80.00, 30.00],
    })
    return {'users': users_data, 'transactions': transactions_data}


@pytest.fixture
def constraint():
    return ForeignToPrimaryKeySubset(
        parent_table_name='users',
        child_table_name='transactions',
        child_foreign_key='user_id',
        conditional_column_name='company_name',
        conditional_values=['TechCorp'],
    )


class TestForeignToPrimaryKeySubset:
    def test___init__(self, constraint):
        """Test the ``__init__`` method sets the parameters."""
        # Assert
        assert constraint.parent_table_name == 'users'
        assert constraint.child_table_name == 'transactions'
        assert constraint.child_foreign_key == 'user_id'
        assert constraint.conditional_column_name == 'company_name'
        assert constraint.conditional_values == ['TechCorp']
        assert constraint._parent_primary_key is None

    def test___init__invalid_parameters(self):
        """Test the ``__init__`` method errors with invalid arguments."""
        # Setup
        parameters = {
            'parent_table_name': 'users',
            'child_table_name': 'transactions',
            'child_foreign_key': 'user_id',
            'conditional_column_name': 'company_name',
            'conditional_values': ['TechCorp'],
        }

        # Run and Assert
        for parameter_name in [
            'parent_table_name',
            'child_table_name',
            'conditional_column_name',
        ]:
            err_msg = re.escape(f'`{parameter_name}` must be a string.')
            with pytest.raises(TypeError, match=err_msg):
                ForeignToPrimaryKeySubset(**{**parameters, parameter_name: 1})

        err_msg = re.escape('`child_foreign_key` must be a string or a list of strings.')
        with pytest.raises(TypeError, match=err_msg):
            ForeignToPrimaryKeySubset(**{**parameters, 'child_foreign_key': 1})

        err_msg = re.escape('`conditional_values` must be a list.')
        with pytest.raises(TypeError, match=err_msg):
            ForeignToPrimaryKeySubset(**{**parameters, 'conditional_values': 'TechCorp'})

    def test__validate_data_missing_table(self, data, metadata, constraint):
        """Test ``_validate_data`` errors if one of the tables is not in the data."""
        # Setup
        del data['users']
        expected_error = re.escape("The table 'users' is missing from the data.")

        # Run and Assert
        with pytest.raises(ConstraintNotApplicableError, match=expected_error):
            constraint._validate_data(data, metadata)

    def test__validate_data_missing_column(self, data, metadata, constraint):
        """Test ``_validate_data`` errors if the conditional column is not in the parent."""
        # Setup
        del data['users']['company_name']
        expected_error = re.escape(
            "The column(s) 'company_name' are missing from the table 'users'."
        )

        # Run and Assert
        with pytest.raises(ConstraintNotApplicableError, match=expected_error):
            constraint._validate_data(data, metadata)

    def test__validate_data_missing_primary_key(self, data, metadata, constraint):
        """Test ``_validate_data`` errors if the parent has no primary key in the metadata."""
        # Setup
        del metadata['tables']['users']['primary_key']
        expected_error = re.escape("The table 'users' does not have a primary key in the metadata.")

        # Run and Assert
        with pytest.raises(ConstraintNotApplicableError, match=expected_error):
            constraint._validate_data(data, metadata)

    def test__validate_data_stores_the_primary_key(self, data, metadata, constraint):
        """Test ``_validate_data`` remembers the primary key that it resolved."""
        # Run
        constraint._validate_data(data, metadata)

        # Assert
        assert constraint._parent_primary_key == 'user_id'

    def test__is_valid(self, data, metadata, constraint):
        """Test ``_is_valid`` considers valid every child row that references a subset row."""
        # Run
        is_valid = constraint._is_valid(data, metadata)

        # Assert
        pd.testing.assert_series_equal(is_valid['transactions'], pd.Series([True] * 10))
        pd.testing.assert_series_equal(is_valid['users'], pd.Series([True] * 10))

    def test__is_valid_with_invalid_values(self, data, metadata, constraint):
        """Test ``_is_valid`` flags a child row that references a row out of the subset."""
        # Setup
        data['transactions']['user_id'] = [1, 2, 3, 8, 1, 3, 8, 1, 3, 8]

        # Run
        is_valid = constraint._is_valid(data, metadata)

        # Assert
        expected = pd.Series([True, False, True, True, True, True, True, True, True, True])
        pd.testing.assert_series_equal(is_valid['transactions'], expected)

    def test__is_valid_with_several_conditional_values(self, data, metadata):
        """Test ``_is_valid`` accepts every conditional value that is allowed."""
        # Setup
        data['transactions']['user_id'] = [1, 2, 3, 8, 1, 3, 8, 1, 3, 8]
        instance = ForeignToPrimaryKeySubset(
            parent_table_name='users',
            child_table_name='transactions',
            child_foreign_key='user_id',
            conditional_column_name='company_name',
            conditional_values=['TechCorp', 'MobileInc'],
        )

        # Run
        is_valid = instance._is_valid(data, metadata)

        # Assert
        pd.testing.assert_series_equal(is_valid['transactions'], pd.Series([True] * 10))

    def test__is_valid_with_unknown_parent(self, data, metadata, constraint):
        """Test ``_is_valid`` accepts an unknown parent while every known one is allowed."""
        # Setup
        data['transactions']['user_id'] = [1, 3, 8, 1, 3, 8, 1, 3, 8, 999]

        # Run
        is_valid = constraint._is_valid(data, metadata)

        # Assert
        pd.testing.assert_series_equal(is_valid['transactions'], pd.Series([True] * 10))

    def test__is_valid_with_unknown_parent_and_invalid_values(self, data, metadata, constraint):
        """Test ``_is_valid`` flags an unknown parent once a known one is out of the subset."""
        # Setup
        data['transactions']['user_id'] = [1, 2, 8, 1, 3, 8, 1, 3, 8, 999]

        # Run
        is_valid = constraint._is_valid(data, metadata)

        # Assert
        expected = pd.Series([True, False, True, True, True, True, True, True, True, False])
        pd.testing.assert_series_equal(is_valid['transactions'], expected)

    def test__is_valid_other_tables(self, data, metadata, constraint):
        """Test ``_is_valid`` considers every row of the other tables valid."""
        # Setup
        data['other_table'] = pd.DataFrame({'a': ['a', 'b']})

        # Run
        is_valid = constraint._is_valid(data, metadata)

        # Assert
        pd.testing.assert_series_equal(is_valid['other_table'], pd.Series([True, True]))

    def test__is_valid_with_another_data(self):
        """Test that all child foreign keys match a primary key that matches the condition."""
        # Setup
        data = {
            'parent_table': pd.DataFrame({
                'parent_pk': [1, 2, 3, 4],
                'conditional_column': ['value_1', 'value_2', 'value_3', 'value_4'],
            }),
            'child_table': pd.DataFrame({'child_fk': [1, 2, 1, 2, 3, 4, 5]}),
        }
        constraint = ForeignToPrimaryKeySubset(
            'parent_table',
            'child_table',
            'child_fk',
            'conditional_column',
            ['value_1', 'value_2'],
        )
        constraint._parent_primary_key = 'parent_pk'

        # Run
        valid_rows = constraint._is_valid(data)

        # Assert
        expected = {
            'parent_table': pd.Series([True, True, True, True]),
            'child_table': pd.Series([True, True, True, True, False, False, False]),
        }
        for table_name, data in expected.items():
            pd.testing.assert_series_equal(data, valid_rows[table_name])

    def test_get_score(self, data, metadata, constraint):
        """Test ``get_score`` returns the proportion of valid rows."""
        # Run and Assert
        assert constraint.get_score(data, metadata) == 1.0

    def test_get_score_invalid_data(self, data, metadata, constraint):
        """Test ``get_score`` counts the rows of every table involved."""
        # Setup
        data['transactions']['user_id'] = [1, 2, 3, 8, 1, 3, 8, 1, 3, 8]

        # Run and Assert
        assert constraint.get_score(data, metadata) == 0.95

    def test_get_score_empty_tables(self, data, metadata, constraint):
        """Test ``get_score`` returns NaN when there are no rows to check."""
        # Setup
        data = {table: table_data.iloc[:0] for table, table_data in data.items()}

        # Run and Assert
        assert pd.isna(constraint.get_score(data, metadata))
