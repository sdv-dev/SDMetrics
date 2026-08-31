import re

import numpy as np
import pandas as pd
import pytest

from sdmetrics.multi_table.statistical.constraints import FixedIncrements
from sdmetrics.multi_table.statistical.constraints.error import ConstraintNotApplicableError


@pytest.fixture
def data():
    return {
        'table': pd.DataFrame({
            'col1': [1, 2, 3, 4],
            'col2': [10, 20, 30, 40],
        })
    }


@pytest.fixture
def metadata():
    return {
        'tables': {
            'table': {
                'columns': {
                    'col1': {'sdtype': 'numerical'},
                    'col2': {'sdtype': 'numerical'},
                }
            }
        }
    }


@pytest.fixture
def constraint():
    return FixedIncrements(
        table_name='table',
        column_name='col1',
        increment_value=1,
    )


class TestFixedIncrements:
    def test___init__invalid_parameters(self):
        """Test ``__init__`` validates the parameter types."""
        # Run and Assert
        err_msg = '`column_name` must be a string.'
        with pytest.raises(ValueError, match=err_msg):
            FixedIncrements(column_name=1, increment_value=10, table_name='table')

        err_msg = 'increment_value` must be an integer or float.'
        with pytest.raises(ValueError, match=err_msg):
            FixedIncrements(column_name='a', increment_value='b', table_name='table')

        err_msg = "The 'table_name' parameter must be a string."
        with pytest.raises(ValueError, match=err_msg):
            FixedIncrements(column_name='a', increment_value=2, table_name=1)

        err_msg = '`increment_value` must be greater than 0.'
        with pytest.raises(ValueError, match=err_msg):
            FixedIncrements(column_name='a', increment_value=-1, table_name='table')

        err_msg = '`increment_value` must be a whole number.'
        with pytest.raises(ValueError, match=err_msg):
            FixedIncrements(column_name='a', increment_value=1.5, table_name='table')

    def test__validate_data_missing_table(self, metadata, constraint):
        """Test ``_validate_data`` errors if the table is not in the data."""
        # Setup
        expected_error = re.escape("The table 'table' is missing from the data.")

        # Run and Assert
        with pytest.raises(ConstraintNotApplicableError, match=expected_error):
            constraint._validate_data({'OtherTable': pd.DataFrame()}, metadata)

    def test__validate_data_missing_columns(self, data, metadata, constraint):
        """Test ``_validate_data`` errors if a referenced column is not in the table."""
        # Setup
        del data['table']['col1']
        expected_error = re.escape("The column 'col1' is missing from the table 'table'.")

        # Run and Assert
        with pytest.raises(ConstraintNotApplicableError, match=expected_error):
            constraint._validate_data(data, metadata)

    def test__is_valid(self, data, metadata, constraint):
        """Test ``_is_valid`` method."""
        # Run
        is_valid = constraint._is_valid(data, metadata)

        # Assert
        expected = pd.Series([True, True, True, True], name='col1')
        pd.testing.assert_series_equal(is_valid['table'], expected)

    @pytest.mark.parametrize(
        'nan',
        [np.nan, pd.NA],
    )
    def test__is_valid_with_different_values(self, nan, metadata):
        """Test it checks if the data is valid."""
        # Setup
        table_name = 'table'
        column_name = 'col1'
        increment_value = 1000
        table_data = {
            table_name: pd.DataFrame({
                column_name: [100, 20000, 55000, 75000, 11000, nan],
                'col2': [23, 42, 34, 13, 40, 12],
            })
        }
        instance = FixedIncrements(
            column_name=column_name, table_name=table_name, increment_value=increment_value
        )

        # Run
        out = instance._is_valid(table_data, metadata)

        # Assert
        expected_out = [False, True, True, True, True, True]
        np.testing.assert_array_equal(expected_out, out[table_name])

    def test_get_score(self, data, metadata, constraint):
        """Test get_score returns the proportion of valid rows."""
        # Run and Assert
        assert constraint.get_score(data, metadata) == 1.0

    def test_get_score_empty_table(self, data, metadata, constraint):
        """Test ``get_score`` returns NaN when there are no rows to check."""
        # Setup
        data['table'] = data['table'].iloc[:0]

        # Run and Assert
        assert pd.isna(constraint.get_score(data, metadata))
