import re

import numpy as np
import pandas as pd
import pytest

from sdmetrics.multi_table.statistical.constraints import ChainedInequality
from sdmetrics.multi_table.statistical.constraints.error import ConstraintNotApplicableError


@pytest.fixture
def data():
    return {
        'table': pd.DataFrame({
            'col_A': [1.0, 2.0, 3.0, 4.0, 5.0],
            'col_B': [2.0, 3.0, 4.0, 5.0, 6.0],
        }),
    }


@pytest.fixture
def metadata():
    return {
        'tables': {
            'table': {
                'columns': {
                    'col_A': {'sdtype': 'numerical'},
                    'col_B': {'sdtype': 'numerical'},
                }
            }
        }
    }


@pytest.fixture
def constraint():
    return ChainedInequality(
        table_name='table',
        column_names=['col_A', 'col_B'],
        strict_boundaries=True,
    )


class TestChainedInequality:
    def test___init__invalid_parameters(self):
        """Test ``__init__`` validates the parameter types."""
        # Run and Assert 1
        with pytest.raises(ValueError, match="The 'table_name' parameter must be a string."):
            ChainedInequality(
                table_name=1,
                column_names=['col_A', 'col_B'],
            )

        # Run and Assert 2
        error_message = "The 'column_names' parameter must be a list of strings."
        with pytest.raises(ValueError, match=error_message):
            ChainedInequality(
                table_name='table',
                column_names=1,
                strict_boundaries=True,
            )

    def test__validate_data_missing_table(self, constraint):
        """Test ``_validate_data`` errors if the table is not in the data."""
        # Setup
        expected_error = re.escape("The table 'table' is missing from the data.")

        # Run and Assert
        with pytest.raises(ConstraintNotApplicableError, match=expected_error):
            constraint._validate_data({'OtherTable': pd.DataFrame()})

    def test__validate_data_missing_columns(self, data, constraint):
        """Test ``_validate_data`` errors if a referenced column is not in the table."""
        # Setup
        del data['table']['col_B']
        expected_error = re.escape("The column(s) 'col_B' are missing from the table 'table'.")

        # Run and Assert
        with pytest.raises(ConstraintNotApplicableError, match=expected_error):
            constraint._validate_data(data)

    def test__is_valid(self, data, metadata, constraint):
        """Test ``_is_valid`` method."""
        # Run
        is_valid = constraint._is_valid(data, metadata)

        # Assert
        expected = pd.Series([True, True, True, True, True])
        pd.testing.assert_series_equal(is_valid['table'], expected)

    def test__is_valid_not_strict(self):
        """Test the ``is_valid`` method with and without strict boundaries."""
        # Setup
        data_float_lower_strict = {
            'table': pd.DataFrame({
                'col_A': [1.0, 2.0, 3.0, 4.0, 5.0],
                'col_B': [2.0, 3.0, 4.0, 5.0, 6.0],
                'col_C': [4.0, 5.0, 6.0, 7.0, 8.0],
                'col_D': [7.0, 8.0, 9.0, 10.0, 11.0],
                'col_E': [11.0, 12.0, 13.0, 14.0, 15.0],
            }),
        }

        data_float_lower_equal = {
            'table': pd.DataFrame({
                'col_A': [1.0, 2.0, 3.0, 4.0, 5.0],
                'col_B': [1.0, 3.0, 4.0, 5.0, 6.0],
                'col_C': [4.0, 3.0, 6.0, 5.0, 8.0],
                'col_D': [7.0, 8.0, 6.0, 5.0, 9.0],
                'col_E': [8.0, 9.0, 9.0, 5.0, 9.0],
            }),
        }
        metadata_float = {
            'tables': {
                'table': {
                    'columns': {
                        'col_A': {'sdtype': 'numerical'},
                        'col_B': {'sdtype': 'numerical'},
                        'col_C': {'sdtype': 'numerical'},
                        'col_D': {'sdtype': 'numerical'},
                        'col_E': {'sdtype': 'numerical'},
                    }
                }
            }
        }

        column_names_strict = data_float_lower_strict['table'].columns.tolist()
        column_names_equal = data_float_lower_equal['table'].columns.tolist()

        constraint_strict = ChainedInequality(
            column_names_strict, strict_boundaries=True, table_name='table'
        )
        constraint_equal = ChainedInequality(
            column_names_equal, strict_boundaries=False, table_name='table'
        )

        # Run
        result_strict = constraint_strict._is_valid(data_float_lower_strict, metadata_float)
        result_equal = constraint_equal._is_valid(data_float_lower_equal, metadata_float)

        # Assert
        expected_result = pd.Series([True] * 5)
        pd.testing.assert_series_equal(result_strict['table'], expected_result)
        pd.testing.assert_series_equal(result_equal['table'], expected_result)

    def test__is_valid_with_and_without_strict(self):
        """Test the ``is_valid`` method with and without strict boundaries."""
        # Setup
        table_data_valid = {
            'table': pd.DataFrame({
                'col_A': [None, 1, None, 5, 6],
                'col_B': [None, None, 3, None, None],
                'col_C': [None, 2, 4, None, 8],
                'col_D': [None, 4, 7, 6, None],
            }),
        }
        metadata = {
            'tables': {
                'table': {
                    'columns': {
                        'col_A': {'sdtype': 'numerical'},
                        'col_B': {'sdtype': 'numerical'},
                        'col_C': {'sdtype': 'numerical'},
                        'col_D': {'sdtype': 'numerical'},
                    }
                }
            }
        }

        datetime_table_data_valid = {
            'table': pd.DataFrame({
                'col_A': [None, None, '03 Jan 2018', '04 Jan 2018', '05 Jan 2018'],
                'col_B': [None, None, None, '05 Jan 2018', None],
                'col_C': [None, '04 Jan 2018', '05 Jan 2018', None, '07 Jan 2018'],
                'col_D': [None, '05 Jan 2018', '06 Jan 2018', '07 Jan 2018', None],
            }).apply(pd.to_datetime)
        }

        table_data_invalid = {
            'table': pd.DataFrame({
                'col_A': [None, 5, 1, None, 6],
                'col_B': [4, None, 3, None, 5],
                'col_C': [None, None, None, 6, 4],
                'col_D': [3, 2, 1, 4, 3],
            })
        }

        datetime_table_data_invalid = {
            'table': pd.DataFrame({
                'col_A': [None, None, '03 Jan 2018', '04 Jan 2018', '05 Jan 2018'],
                'col_B': ['04 Jan 2018', None, None, '05 Jan 2018', None],
                'col_C': [None, '05 Jan 2018', '01 Jan 2018', None, '07 Jan 2016'],
                'col_D': ['04 Jan 2017', '04 Jan 2018', '06 Jan 2018', '02 Jan 2018', None],
            }).apply(pd.to_datetime)
        }

        column_names = ['col_A', 'col_B', 'col_C', 'col_D']
        instance = ChainedInequality(column_names, table_name='table')

        # Run
        result_valid = instance._is_valid(table_data_valid, metadata)
        result_datetime_valid = instance._is_valid(datetime_table_data_valid, metadata)
        result_invalid = instance._is_valid(table_data_invalid, metadata)
        result_datetime_invalid = instance._is_valid(datetime_table_data_invalid, metadata)

        expected_valid = pd.Series([True] * 5)
        expected_invalid = pd.Series([False] * 5)

        # Assert
        pd.testing.assert_series_equal(result_valid['table'], expected_valid)
        pd.testing.assert_series_equal(result_datetime_valid['table'], expected_valid)
        pd.testing.assert_series_equal(result_invalid['table'], expected_invalid)
        pd.testing.assert_series_equal(result_datetime_invalid['table'], expected_invalid)

    def test__is_valid_with_lower_equal_operator(self):
        """Test the ``is_valid`` with the lower equal operator.

        Here there is equal value in the second row for some columns.
        """
        # Setup
        data = {
            'table': pd.DataFrame({
                'col_A': [1, 3, 3, 5, 5],
                'col_B': [2, 3, 4, 5, 6],
                'col_C': [4, 3, 6, 7, 8],
                'col_D': [7, 3, 9, 10, 11],
                'col_E': [8, 4, 13, 14, 15],
            })
        }
        metadata = {
            'tables': {
                'table': {
                    'columns': {
                        'col_A': {'sdtype': 'numerical'},
                        'col_B': {'sdtype': 'numerical'},
                        'col_C': {'sdtype': 'numerical'},
                        'col_D': {'sdtype': 'numerical'},
                        'col_E': {'sdtype': 'numerical'},
                    }
                }
            }
        }
        instance = ChainedInequality(
            data['table'].columns.tolist(), strict_boundaries=False, table_name='table'
        )

        # Run
        result = instance._is_valid(data, metadata)

        # Assert
        expected_result = pd.Series([True, True, True, True, True])
        pd.testing.assert_series_equal(result['table'], expected_result)

    def test__is_valid_with_nans(self):
        """Test the ``is_valid`` when there are NaNs in the columns."""
        # Setup
        table_data_valid = {
            'table': pd.DataFrame({
                'low': [1, np.nan, 3, 4, np.nan, 1],
                'middle': [2, 3, np.nan, 5, np.nan, np.nan],
                'high': [3, 4, 5, np.nan, 6, np.nan],
            })
        }
        table_data_invalid = {
            'table': pd.DataFrame({
                'low': [1, np.nan, 3, 4, np.nan, 1],
                'middle': [2, 3, np.nan, 5, np.nan, np.nan],
                'high': [3, 4, 2, np.nan, 6, np.nan],
            })
        }
        metadata = {
            'tables': {
                'table': {
                    'columns': {
                        'low': {'sdtype': 'numerical'},
                        'middle': {'sdtype': 'numerical'},
                        'high': {'sdtype': 'numerical'},
                    }
                }
            }
        }
        instance = ChainedInequality(
            column_names=['low', 'middle', 'high'],
            strict_boundaries=False,
            table_name='table',
        )

        # Run
        result_valid = instance._is_valid(table_data_valid, metadata)
        result_invalid = instance._is_valid(table_data_invalid, metadata)

        expected_valid = pd.Series([True] * 6)
        expected_invalid = pd.Series([True, True, False, True, True, True])

        # Assert
        pd.testing.assert_series_equal(result_valid['table'], expected_valid)
        pd.testing.assert_series_equal(result_invalid['table'], expected_invalid)

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
