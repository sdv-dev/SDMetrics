import re

import numpy as np
import pandas as pd
import pytest

from sdmetrics.multi_table.statistical.constraints import DenormalizedTable
from sdmetrics.multi_table.statistical.constraints.error import ConstraintNotApplicableError


@pytest.fixture
def data():
    return {
        'tableA': pd.DataFrame({
            'id': [1, 1, 2, 2, 3],
            'dob': ['1990-01-01', '1990-01-01', '1985-05-05', '1985-05-05', '1970-02-02'],
            'name': ['Ann', 'Ann', 'Bob', 'Bob', 'Cam'],
            'last_name': ['A', 'A', 'B', 'B', 'C'],
            'amount': [10, 20, 30, 40, 50],
        })
    }


@pytest.fixture
def constraint():
    return DenormalizedTable(
        table_name='tableA',
        denormalized_primary_key='id',
        denormalized_column_names=['dob', 'name', 'last_name'],
    )


class TestDenormalizedTable:
    def test___init__invalid_parameters(self):
        """Test ``__init__`` validates the parameter types."""
        # Run & Assert 1
        with pytest.raises(ValueError, match="The 'table_name' parameter must be a string."):
            DenormalizedTable(table_name=1, denormalized_primary_key='id')

        # Run & Assert 2
        error_message = "The 'denormalized_column_names' parameter must be a list of strings."
        with pytest.raises(ValueError, match=error_message):
            DenormalizedTable(
                table_name='tableA',
                denormalized_primary_key='id',
                denormalized_column_names='name',
            )

    def test___init__primary_key_in_column_names(self):
        """Test ``__init__`` errors if the primary key is also a denormalized column."""
        # Run & Assert
        with pytest.raises(ValueError, match='cannot be both'):
            DenormalizedTable(
                table_name='tableA',
                denormalized_primary_key='id',
                denormalized_column_names=['id'],
            )

    def test__validate_data_missing_table(self, constraint):
        """Test ``_validate_data`` errors if the table is not in the data."""
        # Setup
        expected_error = re.escape("The table 'tableA' is missing from the data.")

        # Run & Assert
        with pytest.raises(ConstraintNotApplicableError, match=expected_error):
            constraint._validate_data({'OtherTable': pd.DataFrame()})

    def test__validate_data_missing_columns(self, data, constraint):
        """Test ``_validate_data`` errors if a referenced column is not in the table."""
        # Setup
        del data['tableA']['name']
        expected_error = re.escape("The column(s) 'name' are missing from the table 'tableA'.")

        # Run & Assert
        with pytest.raises(ConstraintNotApplicableError, match=expected_error):
            constraint._validate_data(data)

    def test__is_valid(self, data, constraint):
        """Test ``_is_valid`` method."""
        # Run
        is_valid = constraint._is_valid(data)

        # Assert
        expected = pd.Series([True, True, True, True, True])
        pd.testing.assert_series_equal(is_valid['tableA'], expected)

    def test__is_valid_no_denormalized_columns(self, data):
        """Test ``_is_valid`` return every row is valid when there is nothing to check."""
        # Setup
        instance = DenormalizedTable(table_name='tableA', denormalized_primary_key='id')

        # Run
        is_valid = instance._is_valid(data)

        # Assert
        pd.testing.assert_series_equal(is_valid['tableA'], pd.Series([True] * 5))

    def test__is_valid_with_nans(self, constraint):
        """Test ``_is_valid`` missing values are treated as equal to each other."""
        # Setup
        data = {
            'tableA': pd.DataFrame({
                'id': [1, 1, None, None],
                'dob': [np.nan, np.nan, '1970-02-02', '1970-02-02'],
                'name': ['Ann', 'Ann', 'Cam', 'Cam'],
                'last_name': ['A', 'A', 'C', 'C'],
            })
        }

        # Run
        is_valid = constraint._is_valid(data)

        # Assert
        pd.testing.assert_series_equal(is_valid['tableA'], pd.Series([True] * 4))

    def test__is_valid_empty_table(self, data, constraint):
        """Test it returns all true when the table is empty."""
        # Setup
        empty = data['tableA'].iloc[0:0].copy()
        data['tableA'] = empty

        # Run
        valid_rows = constraint._is_valid(data)

        # Assert
        assert valid_rows['tableA'].empty

    def test__is_valid_inconsistent_other_denorm_column(self, data, constraint):
        """Variation in any denormalized column marks all rows for that key invalid."""
        # Setup
        data['tableA'].loc[3, 'name'] = 'Cam'

        # Run
        valid_rows = constraint._is_valid(data)

        # Assert
        expected = pd.Series([True, True, False, False, True])
        pd.testing.assert_series_equal(valid_rows['tableA'], expected)

    def test_get_score(self, data, constraint):
        """Test get_score returns the proportion of valid rows."""
        # Setup
        data['tableA'].loc[3, 'name'] = 'Cam'

        # Run & Assert
        assert constraint.get_score(data) == 0.6

    def test_get_score_empty_table(self, data, constraint):
        """Test ``get_score`` returns NaN when there are no rows to check."""
        # Setup
        data['tableA'] = data['tableA'].iloc[:0]

        # Run & Assert
        assert pd.isna(constraint.get_score(data))
