import re
from datetime import datetime

import numpy as np
import pandas as pd
import pytest

from sdmetrics.multi_table.statistical.constraints import Inequality
from sdmetrics.multi_table.statistical.constraints.error import ConstraintNotApplicableError


@pytest.fixture
def data():
    return {
        'table': pd.DataFrame({
            'col1': [1, 2, 3, 4],
            'col2': [10, 20, 30, 40],
            'col3': [1, 20, 300, 4000],
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
                    'col3': {'sdtype': 'numerical'},
                }
            }
        }
    }


@pytest.fixture
def constraint():
    return Inequality(
        table_name='table',
        low_column_name='col1',
        high_column_name='col2',
        strict_boundaries=True,
    )


class TestInequality:
    def test___init__invalid_parameters(self):
        """Test ``__init__`` validates the parameter types."""
        # Run & Assert 1
        with pytest.raises(ValueError, match="The 'table_name' parameter must be a string."):
            Inequality(
                table_name=1,
                low_column_name='col1',
                high_column_name='col2',
                strict_boundaries=True,
            )

        # Run & Assert 2
        error_message = (
            '`low_column_name` and `high_column_name` must be strings.'
        )
        with pytest.raises(ValueError, match=error_message):
            Inequality(
                table_name='table',
                low_column_name='col1',
                high_column_name=1,
                strict_boundaries=True,
            )

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
        del data['table']['col2']
        expected_error = re.escape("The column(s) 'col2' are missing from the table 'table'.")

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

    def test__is_valid_checks_data_is_valid(self):
        """Test ``_is_valid`` checks if the data is valid."""
        # Setup
        table_data = {
            'table': pd.DataFrame({
                'a': [1, np.nan, 3, 4, None, 6, 8, 0],
                'b': [4, 2, 2, 4, np.nan, -6, 10, float('nan')],
                'col': [7, 8, 9, 10, 11, 12, 13, 14],
            })
        }
        metadata_dict = {
            'tables': {
                'table': {
                    'columns': {
                        'a': {'sdtype': 'numerical'},
                        'b': {'sdtype': 'numerical'},
                        'col': {'sdtype': 'numerical'},
                    }
                }
            }
        }
        instance = Inequality(
            low_column_name='a',
            high_column_name='b',
            strict_boundaries=True,
            table_name='table',
        )

        # Run
        out = instance._is_valid(table_data, metadata_dict)

        # Assert
        out = out['table']
        expected_out = [True, True, False, False, True, False, True, True]
        np.testing.assert_array_equal(expected_out, out)

    def test_is_valid_strict_boundaries_true(self):
        """Test it checks if the data is valid when strict boundaries are False."""
        # Setup
        table_data = {
            'table': pd.DataFrame({
                'a': [1, np.nan, 3, 3, None, 6, 8, 0],
                'b': [4, 2, 2, 4, np.nan, -6, 10, float('nan')],
                'col': [7, 8, 9, 10, 11, 12, 13, 14],
            })
        }
        metadata_dict = {
            'tables': {
                'table': {
                    'columns': {
                        'a': {'sdtype': 'numerical'},
                        'b': {'sdtype': 'numerical'},
                        'col': {'sdtype': 'numerical'},
                    }
                }
            }
        }
        instance = Inequality(
            low_column_name='a',
            high_column_name='b',
            strict_boundaries=False,
            table_name='table',
        )

        # Run
        out = instance._is_valid(table_data, metadata_dict)

        # Assert
        out = out['table']
        expected_out = [True, True, False, True, True, False, True, True]
        np.testing.assert_array_equal(expected_out, out)

    def test_is_valid_datetimes(self):
        """Test it checks if the data is valid when it contains datetimes."""
        # Setup
        table_data = {
            'table': pd.DataFrame({
                'a': ['2020-5-17', '2021-9-1', None],
                'b': [datetime(2020, 5, 18), datetime(2020, 9, 2), datetime(2020, 9, 2)],
                'c': [datetime(2020, 5, 29), datetime(2021, 9, 3), np.nan],
                'col': [7, 8, 9],
            })
        }
        metadata_dict = {
            'tables': {
                'table': {
                    'columns': {
                        'a': {'sdtype': 'datetime'},
                        'b': {'sdtype': 'datetime'},
                        'col': {'sdtype': 'numerical'},
                    }
                }
            }
        }
        instance = Inequality(
            low_column_name='a',
            high_column_name='b',
            strict_boundaries=True,
            table_name='table',
        )

        # Run
        out = instance._is_valid(table_data, metadata_dict)

        # Assert
        out = out['table']
        expected_out = [True, False, True]
        np.testing.assert_array_equal(expected_out, out)

    def test_is_valid_datetimes_strings(self):
        """Test it checks if the data is valid when it contains datetimes."""
        # Setup
        table_data = {
            'table': pd.DataFrame({
                'a': ['2020-05-17', '2021-09-01', '2021-09-01'],
                'b': ['2020-05-18', '2020-09-02', '2021-09-02'],
                'c': ['2020-05-29', '2021-09-03', '2021-09-03'],
                'col': [7, 8, 9],
            })
        }
        metadata_dict = {
            'tables': {
                'table': {
                    'columns': {
                        'a': {'sdtype': 'datetime'},
                        'b': {'sdtype': 'datetime'},
                        'col': {'sdtype': 'numerical'},
                    }
                }
            }
        }
        instance = Inequality(
            low_column_name='a',
            high_column_name='b',
            table_name='table',
            strict_boundaries=True,
        )

        # Run
        out = instance._is_valid(table_data, metadata_dict)

        # Assert
        out = out['table']
        expected_out = [True, False, True]
        np.testing.assert_array_equal(expected_out, out)

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
