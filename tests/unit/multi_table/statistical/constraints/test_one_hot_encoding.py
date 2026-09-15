import re

import numpy as np
import pandas as pd
import pytest

from sdmetrics.multi_table.statistical.constraints import OneHotEncoding
from sdmetrics.multi_table.statistical.constraints.error import ConstraintNotApplicableError


@pytest.fixture
def data():
    return {
        'tableA': pd.DataFrame({
            'col1': [1, 0, 0, 1],
            'col2': [0, 1, 0, 0],
            'col3': [0, 0, 1, 0],
            'col4': [10, 20, 30, 40],
        })
    }


@pytest.fixture
def constraint():
    return OneHotEncoding(
        table_name='tableA',
        column_names=['col1', 'col2', 'col3'],
    )


class TestOneHotEncoding:
    def test___init__invalid_parameters(self):
        """Test ``__init__`` validates the parameter types."""
        # Run and Assert 1
        with pytest.raises(ValueError, match="The 'table_name' parameter must be a string."):
            OneHotEncoding(table_name=1, column_names=['col1', 'col2', 'col3'])

        # Run and Assert 2
        error_message = "The 'column_names' parameter must be a list of strings."
        with pytest.raises(ValueError, match=error_message):
            OneHotEncoding(
                table_name='tableA',
                column_names='col1',
            )

    def test__validate_data_missing_table(self, constraint):
        """Test ``_validate_data`` errors if the table is not in the data."""
        # Setup
        expected_error = re.escape("The table 'tableA' is missing from the data.")

        # Run and Assert
        with pytest.raises(ConstraintNotApplicableError, match=expected_error):
            constraint._validate_data({'OtherTable': pd.DataFrame()})

    def test__validate_data_missing_columns(self, data, constraint):
        """Test ``_validate_data`` errors if a referenced column is not in the table."""
        # Setup
        del data['tableA']['col2']
        expected_error = re.escape("The column(s) 'col2' are missing from the table 'tableA'.")

        # Run and Assert
        with pytest.raises(ConstraintNotApplicableError, match=expected_error):
            constraint._validate_data(data)

    def test__is_valid(self, data, constraint):
        """Test ``_is_valid`` method."""
        # Run
        is_valid = constraint._is_valid(data)

        # Assert
        expected = pd.Series([True, True, True, True])
        pd.testing.assert_series_equal(is_valid['tableA'], expected)

    def test__is_valid_with_incomplete_score(self):
        """Test ``_is_valid`` does not completely pass."""
        # Setup
        instance = OneHotEncoding(column_names=['a', 'b', 'c'], table_name='table')

        # Run
        table_data = pd.DataFrame({
            'a': [1.0, 1.0, 0.0, 0.5, 1.0],
            'b': [0.0, 1.0, 0.0, 0.5, 0.0],
            'c': [0.0, 2.0, 0.0, 0.0, np.nan],
            'd': [1, 2, 3, 4, 5],
        })
        data = {'table': table_data, 'table2': table_data}
        out = instance.is_valid(data)

        # Assert
        expected_out = {
            'table': pd.Series([True, False, False, False, False]),
            'table2': pd.Series([True, True, True, True, True]),
        }
        for table, series in out.items():
            pd.testing.assert_series_equal(expected_out[table], series)

    def test_get_score(self, data, constraint):
        """Test get_score returns the proportion of valid rows."""
        # Run and Assert
        assert constraint.get_score(data) == 1.0

    def test_get_score_empty_table(self, data, constraint):
        """Test ``get_score`` returns NaN when there are no rows to check."""
        # Setup
        data['tableA'] = data['tableA'].iloc[:0]

        # Run and Assert
        assert pd.isna(constraint.get_score(data))
