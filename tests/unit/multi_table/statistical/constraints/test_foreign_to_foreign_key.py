import re

import pandas as pd
import pytest

from sdmetrics.multi_table.statistical.constraints import ForeignToForeignKey
from sdmetrics.multi_table.statistical.constraints.error import ConstraintNotApplicableError


@pytest.fixture
def metadata():
    return {
        'tables': {
            'users': {
                'primary_key': 'user_id',
                'columns': {
                    'user_id': {'sdtype': 'id'},
                    'product_id': {'sdtype': 'id'},
                    'company_name': {'sdtype': 'company', 'pii': True},
                    'user_name': {'sdtype': 'name', 'pii': True},
                },
            },
            'transactions': {
                'primary_key': 'transaction_id',
                'columns': {
                    'transaction_id': {'sdtype': 'id'},
                    'product_id': {'sdtype': 'id'},
                    'company_name': {'sdtype': 'company', 'pii': True},
                    'amount': {'sdtype': 'unknown', 'pii': True},
                },
            },
        }
    }


@pytest.fixture
def data():
    users_data = pd.DataFrame({
        'user_id': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
        'product_id': [101, 102, 101, 103, 104, 105, 106, 101, 102, 103],
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
        'product_id': [101, 102, 101, 103, 104, 105, 106, 101, 102, 103],
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
        'amount': [50.00, 75.00, 20.00, 60.00, 90.00, 100.00, 150.00, 55.00, 80.00, 30.00],
    })
    return {'users': users_data, 'transactions': transactions_data}


@pytest.fixture
def columns():
    return [
        {'table_name': 'users', 'foreign_key': 'product_id'},
        {'table_name': 'transactions', 'foreign_key': 'product_id'},
    ]


@pytest.fixture
def constraint(columns):
    return ForeignToForeignKey(columns=columns)


class TestForeignToForeignKey:
    def test___init__(self, columns):
        """Test the ``__init__`` method sets the parameters."""
        # Run
        instance = ForeignToForeignKey(columns=columns)

        # Assert
        assert instance.columns == columns
        assert instance.foreign_key_generation == 'new'

    def test___init__with_composite_keys(self):
        """Test the ``__init__`` method accepts composite foreign keys."""
        # Setup
        columns = [
            {'table_name': 'users', 'foreign_key': ('product_id', 'company_name')},
            {'table_name': 'transactions', 'foreign_key': ('product_id', 'company_name')},
        ]

        # Run
        instance = ForeignToForeignKey(columns=columns, foreign_key_generation='reuse')

        # Assert
        assert instance.columns == columns
        assert instance.foreign_key_generation == 'reuse'

    def test___init__invalid_columns(self, columns):
        """Test the ``__init__`` method errors if ``columns`` is malformed."""
        # Run and Assert
        err_msg = re.escape('columns must be a list of dictionaries')
        with pytest.raises(ValueError, match=err_msg):
            ForeignToForeignKey(columns=columns[0])

        err_msg = re.escape('Each entry in columns must be a dictionary')
        with pytest.raises(ValueError, match=err_msg):
            ForeignToForeignKey(columns=[columns[0], 'transactions'])

    def test___init__invalid_entry_keys(self, columns):
        """Test the ``__init__`` method errors if an entry has the wrong keys."""
        # Run and Assert
        err_msg = re.escape("Each dictionary must have a 'table_name' key with a string value")
        with pytest.raises(ValueError, match=err_msg):
            ForeignToForeignKey(columns=[columns[0], {'foreign_key': 'product_id'}])

        with pytest.raises(ValueError, match=err_msg):
            ForeignToForeignKey(columns=[columns[0], {**columns[1], 'table_name': 1}])

        err_msg = re.escape("Each dictionary must have a 'foreign_key' key")
        with pytest.raises(ValueError, match=err_msg):
            ForeignToForeignKey(columns=[columns[0], {'table_name': 'transactions'}])

        err_msg = re.escape("'foreign_key' must be a string or a tuple of strings")
        with pytest.raises(ValueError, match=err_msg):
            ForeignToForeignKey(columns=[columns[0], {**columns[1], 'foreign_key': ['a']}])

    def test___init__mismatched_composite_keys(self, columns):
        """Test the ``__init__`` method errors if the keys do not have the same size."""
        # Setup
        composite_info = {**columns[1], 'foreign_key': ('product_id', 'company_name')}
        err_msg = re.escape(
            'All foreign key entries must have the same number of columns. Entry for table '
            "'transactions' has 2 columns, expected 1."
        )

        # Run and Assert
        with pytest.raises(ValueError, match=err_msg):
            ForeignToForeignKey(columns=[columns[0], composite_info])

    def test___init__invalid_foreign_key_generation(self, columns):
        """Test the ``__init__`` method errors with an unknown foreign key generation."""
        # Run and Assert
        err_msg = re.escape('`foreign_key_generation` must be a string.')
        with pytest.raises(ValueError, match=err_msg):
            ForeignToForeignKey(columns=columns, foreign_key_generation=1)

        err_msg = re.escape(
            "Unrecognized `foreign_key_generation` value 'copy'. Must be one of ['new', 'reuse']."
        )
        with pytest.raises(ValueError, match=err_msg):
            ForeignToForeignKey(columns=columns, foreign_key_generation='copy')

    def test__validate_data_missing_table(self, data, metadata, constraint):
        """Test ``_validate_data`` errors if one of the tables is not in the data."""
        # Setup
        del data['transactions']
        expected_error = re.escape("The table 'transactions' is missing from the data.")

        # Run & Assert
        with pytest.raises(ConstraintNotApplicableError, match=expected_error):
            constraint._validate_data(data, metadata)

    def test__validate_data_missing_column(self, data, metadata, constraint):
        """Test ``_validate_data`` errors if a foreign key column is not in the table."""
        # Setup
        del data['users']['product_id']
        expected_error = re.escape("The column(s) 'product_id' are missing from the table 'users'.")

        # Run & Assert
        with pytest.raises(ConstraintNotApplicableError, match=expected_error):
            constraint._validate_data(data, metadata)

    def test__validate_data_missing_composite_column(self, data, metadata):
        """Test ``_validate_data`` checks every column of a composite foreign key."""
        # Setup
        instance = ForeignToForeignKey(
            columns=[
                {'table_name': 'users', 'foreign_key': ('product_id', 'company_name')},
                {'table_name': 'transactions', 'foreign_key': ('product_id', 'company_name')},
            ]
        )
        del data['transactions']['company_name']
        expected_error = re.escape(
            "The column(s) 'company_name' are missing from the table 'transactions'."
        )

        # Run & Assert
        with pytest.raises(ConstraintNotApplicableError, match=expected_error):
            instance._validate_data(data, metadata)

    def test__is_valid(self, data, metadata, constraint):
        """Test the ``_is_valid`` method returns True for all rows."""
        # Run
        is_valid = constraint._is_valid(data, metadata)

        # Assert
        assert set(is_valid) == {'users', 'transactions'}
        pd.testing.assert_series_equal(is_valid['users'], pd.Series([True] * 10))
        pd.testing.assert_series_equal(is_valid['transactions'], pd.Series([True] * 10))

    def test__is_valid_with_unshared_values(self, data, metadata, constraint):
        """Test ``_is_valid`` also accepts a value that only one of the tables holds."""
        # Setup
        data['users'].loc[2, 'product_id'] = 999

        # Run
        is_valid = constraint._is_valid(data, metadata)

        # Assert
        pd.testing.assert_series_equal(is_valid['users'], pd.Series([True] * 10))
        pd.testing.assert_series_equal(is_valid['transactions'], pd.Series([True] * 10))

    def test__is_valid_other_tables(self, data, metadata, constraint):
        """Test ``_is_valid`` considers every row of the other tables valid."""
        # Setup
        data['other_table'] = pd.DataFrame({'a': ['a', 'b']})

        # Run
        is_valid = constraint._is_valid(data, metadata)

        # Assert
        pd.testing.assert_series_equal(is_valid['other_table'], pd.Series([True, True]))

    def test_get_score(self, data, metadata, constraint):
        """Test ``get_score`` returns the proportion of valid rows."""
        # Run & Assert
        assert constraint.get_score(data, metadata) == 1.0

    def test_get_score_with_unshared_values(self, data, metadata, constraint):
        """Test ``get_score`` stays at one when the tables do not share a value."""
        # Setup
        data['users'].loc[2, 'product_id'] = 999

        # Run & Assert
        assert constraint.get_score(data, metadata) == 1.0

    def test_get_score_empty_tables(self, data, metadata, constraint):
        """Test ``get_score`` returns NaN when there are no rows to check."""
        # Setup
        data = {table: table_data.iloc[:0] for table, table_data in data.items()}

        # Run & Assert
        assert pd.isna(constraint.get_score(data, metadata))
