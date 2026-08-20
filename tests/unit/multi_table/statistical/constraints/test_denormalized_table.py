import re

import numpy as np
import pandas as pd
import pytest

from sdmetrics.multi_table.statistical.constraints import DenormalizedTable
from sdmetrics.multi_table.statistical.constraints.base import ConstraintNotApplicableError


@pytest.fixture
def data():
    return {
        'UserTransaction': pd.DataFrame({
            'User ID': [1, 1, 2, 2, 3],
            'DateOfBirth': ['1990-01-01', '1990-01-01', '1985-05-05', '1985-05-05', '1970-02-02'],
            'FirstName': ['Ann', 'Ann', 'Bob', 'Bob', 'Cam'],
            'LastName': ['A', 'A', 'B', 'B', 'C'],
            'Amount': [10, 20, 30, 40, 50],
        })
    }


@pytest.fixture
def constraint():
    return DenormalizedTable(
        table_name='UserTransaction',
        denormalized_primary_key='User ID',
        denormalized_column_names=['DateOfBirth', 'FirstName', 'LastName'],
    )


class TestDenormalizedTable:
    def test___init__invalid_parameters(self):
        """Test ``__init__`` validates the parameter types."""
        # Run & Assert 1
        with pytest.raises(ValueError, match="The 'table_name' parameter must be a string."):
            DenormalizedTable(table_name=1, denormalized_primary_key='User ID')

        # Run & Assert 2
        error_message = "The 'denormalized_column_names' parameter must be a list of strings."
        with pytest.raises(ValueError, match=error_message):
            DenormalizedTable(
                table_name='UserTransaction',
                denormalized_primary_key='User ID',
                denormalized_column_names='FirstName',
            )

    def test___init__primary_key_in_column_names(self):
        """Test ``__init__`` errors if the primary key is also a denormalized column."""
        # Run & Assert
        with pytest.raises(ValueError, match='cannot be both'):
            DenormalizedTable(
                table_name='UserTransaction',
                denormalized_primary_key='User ID',
                denormalized_column_names=['User ID'],
            )

    def test__validate_data_missing_table(self, constraint):
        """Test ``_validate_data`` errors if the table is not in the data."""
        # Setup
        expected_error = re.escape("The table 'UserTransaction' is missing from the data.")

        # Run & Assert
        with pytest.raises(ConstraintNotApplicableError, match=expected_error):
            constraint._validate_data({'OtherTable': pd.DataFrame()})

    def test__validate_data_missing_columns(self, data, constraint):
        """Test ``_validate_data`` errors if a referenced column is not in the table."""
        # Setup
        del data['UserTransaction']['FirstName']
        expected_error = re.escape(
            "The column(s) 'FirstName' are missing from the table 'UserTransaction'."
        )

        # Run & Assert
        with pytest.raises(ConstraintNotApplicableError, match=expected_error):
            constraint._validate_data(data)

    def test__is_valid(self, data, constraint):
        """Test ``_is_valid`` method."""
        # Run
        is_valid = constraint._is_valid(data)

        # Assert
        expected = pd.Series([True, True, True, True, True])
        pd.testing.assert_series_equal(is_valid['UserTransaction'], expected)

    def test__is_valid_no_denormalized_columns(self, data):
        """Test ``_is_valid`` return every row is valid when there is nothing to check."""
        # Setup
        instance = DenormalizedTable(
            table_name='UserTransaction', denormalized_primary_key='User ID'
        )

        # Run
        is_valid = instance._is_valid(data)

        # Assert
        pd.testing.assert_series_equal(is_valid['UserTransaction'], pd.Series([True] * 5))

    def test__is_valid_with_nans(self, constraint):
        """Test ``_is_valid`` missing values are treated as equal to each other."""
        # Setup
        data = {
            'UserTransaction': pd.DataFrame({
                'User ID': [1, 1, None, None],
                'DateOfBirth': [np.nan, np.nan, '1970-02-02', '1970-02-02'],
                'FirstName': ['Ann', 'Ann', 'Cam', 'Cam'],
                'LastName': ['A', 'A', 'C', 'C'],
            })
        }

        # Run
        is_valid = constraint._is_valid(data)

        # Assert
        pd.testing.assert_series_equal(is_valid['UserTransaction'], pd.Series([True] * 4))

    def test__is_valid_empty_table(self, data, constraint):
        """Test it returns all true when the table is empty."""
        # Setup
        empty = data['UserTransaction'].iloc[0:0].copy()
        data['UserTransaction'] = empty

        # Run
        valid_rows = constraint._is_valid(data)

        # Assert
        assert valid_rows['UserTransaction'].empty

    def test__is_valid_inconsistent_other_denorm_column(self, data, constraint):
        """Variation in any denormalized column marks all rows for that key invalid."""
        # Setup
        data['UserTransaction'].loc[3, 'FirstName'] = 'Cam'

        # Run
        valid_rows = constraint._is_valid(data)

        # Assert
        expected = pd.Series([True, True, False, False, True])
        pd.testing.assert_series_equal(valid_rows['UserTransaction'], expected)

    def test_get_score(self, data, constraint):
        """Test get_score returns the proportion of valid rows."""
        # Setup
        data['UserTransaction'].loc[3, 'FirstName'] = 'Cam'

        # Run & Assert
        assert constraint.get_score(data) == 0.6

    def test_get_score_empty_table(self, data, constraint):
        """Test ``get_score`` returns NaN when there are no rows to check."""
        # Setup
        data['UserTransaction'] = data['UserTransaction'].iloc[:0]

        # Run & Assert
        assert np.isnan(constraint.get_score(data))
