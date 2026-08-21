import re
from copy import deepcopy

import numpy as np
import pandas as pd
import pytest

from sdmetrics.multi_table.statistical.constraints import FixedCombinations
from sdmetrics.multi_table.statistical.constraints.error import ConstraintNotApplicableError


@pytest.fixture
def data():
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
    return FixedCombinations(
        table_name='table',
        column_names=['a', 'b', 'c']
    )


class TestFixedCombinations:
    def test___init__invalid_parameters(self):
        """Test the ``__init__`` method errors with invalid arguments."""
        # Setup
        bad_column_names_type = 'col1'
        bad_column_names = ['col1', 2]
        short_column_names = ['col1']
        bad_table_name = 1

        # Run and assert
        bad_column_names_msg = re.escape('`column_names` must be a list of strings.')
        with pytest.raises(ValueError, match=bad_column_names_msg):
            FixedCombinations(bad_column_names_type, table_name='table')

        with pytest.raises(ValueError, match=bad_column_names_msg):
            FixedCombinations(bad_column_names, table_name='table')

        short_column_names_msg = re.escape(
            'FixedCombinations constraint requires at least two columns.'
        )
        with pytest.raises(ValueError, match=short_column_names_msg):
            FixedCombinations(short_column_names, table_name='table')

        bad_table_name_msg = re.escape('`table_name` must be a string or None.')
        with pytest.raises(ValueError, match=bad_table_name_msg):
            FixedCombinations(column_names=['col1', 'col2'], table_name=bad_table_name)

    def test__validate_data_missing_table(self, constraint):
        """Test ``_validate_data`` errors if the table is not in the data."""
        # Setup
        expected_error = re.escape("The table 'table' is missing from the data.")

        # Run & Assert
        with pytest.raises(ConstraintNotApplicableError, match=expected_error):
            constraint._validate_data({'OtherTable': pd.DataFrame()})

    def test__validate_data_missing_columns(self, data, constraint):
        """Test ``_validate_data`` errors if a referenced column is not in the table."""
        # Setup
        del data['table']['a']
        expected_error = re.escape("The column 'a' is missing from the table 'table'.")

        # Run & Assert
        with pytest.raises(ConstraintNotApplicableError, match=expected_error):
            constraint._validate_data(data)

    def test__is_valid(self, data, metadata, constraint):
        """Test ``_is_valid`` method."""
        # Run
        is_valid = constraint._is_valid(data, metadata)

        # Assert
        expected = pd.Series([True, True, True, True])
        pd.testing.assert_series_equal(is_valid['table'], expected)

    def test__is_valid_with_invalid_values(self, data):
        """Test the ``_is_valid`` with invalid data."""
        # Setup
        invalid_data = pd.DataFrame({
            'a': ['a', 'b', 'c'],
            'b': ['D', 'E', 'F'],
            'c': ['g', 'h', 'i'],
        })

        columns = ['b', 'c']
        instance = FixedCombinations(column_names=columns, table_name='table')

        # Run
        valid_out = instance._is_valid(data)
        invalid_out = instance._is_valid(invalid_data)

        # Assert
        expected_valid_out = pd.Series([True, True, True], name='b#c')
        pd.testing.assert_series_equal(expected_valid_out, valid_out)
        pd.testing.assert_series_equal(~expected_valid_out, invalid_out)

    def test__is_valid_non_string(self):
        """Test the ``_is_valid`` with non-string input."""
        # Setup
        data = pd.DataFrame({
            'a': ['a', 'b', 'c'],
            'b': [1, 2, 3],
            'c': ['g', 'h', 'i'],
            'd': [2.4, 1.23, 5.6],
        })
        invalid_data = pd.DataFrame({
            'a': ['a', 'b', 'c'],
            'b': [6, 7, 8],
            'c': ['g', 'h', 'i'],
            'd': [2.4, 1.23, 5.6],
        })

        columns = ['b', 'c', 'd']
        instance = FixedCombinations(column_names=columns, table_name='table')

        # Run
        valid_out = instance._is_valid(data)
        invalid_out = instance._is_valid(invalid_data)

        # Assert
        expected_valid_out = pd.Series([True, True, True], name='b#c#d')
        pd.testing.assert_series_equal(expected_valid_out, valid_out)
        pd.testing.assert_series_equal(~expected_valid_out, invalid_out)

    def test__is_valid_with_nans(self, data, metadata):
        """Test the ``FixedCombinations.is_valid`` method."""
        # Setup
        data = pd.DataFrame({
            'a': ['a', 'b', 'c', 'g', 'k', 'l'],
            'b': ['d', 'e', 'f', None, np.nan, 'f'],
            'c': ['g', 'h', None, None, None, None],
            'd': [2.4, 1.23, 5.6, 4.5, 3.2, 5.6],
        })
        invalid_data = pd.DataFrame({
            'a': ['a', 'b', 'c'],
            'b': ['D', 'E', 'F'],
            'c': ['g', 'h', 'i'],
        })

        columns = ['b', 'c']
        instance = FixedCombinations(column_names=columns, table_name='table')

        # Run
        valid_out = instance._is_valid(data)
        invalid_out = instance._is_valid(invalid_data)

        # Assert
        expected_valid_out = pd.Series([True] * 6, name='b#c')
        pd.testing.assert_series_equal(expected_valid_out, valid_out)

        expected_invalid_out = pd.Series([False] * 3, name='b#c')
        pd.testing.assert_series_equal(expected_invalid_out, invalid_out)


    def test_get_score(self, data, metadata, constraint):
        """Test get_score returns the proportion of valid rows."""
        # Run & Assert
        assert constraint.get_score(data, metadata) == 1.0

    def test_get_score_empty_table(self, data, metadata, constraint):
        """Test ``get_score`` returns NaN when there are no rows to check."""
        # Setup
        data['table'] = data['table'].iloc[:0]

        # Run & Assert
        assert pd.isna(constraint.get_score(data, metadata))
