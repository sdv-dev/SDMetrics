import re

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
    return FixedCombinations(table_name='table', column_names=['a', 'b', 'c'])


class TestFixedCombinations:
    def test___init__invalid_parameters(self):
        """Test the ``__init__`` method errors with invalid arguments."""
        # Run and Assert
        err_msg = "The 'table_name' parameter must be a string."
        with pytest.raises(ValueError, match=err_msg):
            FixedCombinations(column_names=['a', 'b'], table_name=1)

        err_msg = re.escape("The 'column_names' parameter must be a list of strings.")
        with pytest.raises(ValueError, match=err_msg):
            FixedCombinations(column_names='a', table_name='table')

        with pytest.raises(ValueError, match=err_msg):
            FixedCombinations(column_names=['a', 2], table_name='table')

        err_msg = re.escape("FixedCombinations constraint requires at least two columns.")
        with pytest.raises(ValueError, match=err_msg):
            FixedCombinations(column_names=['a'], table_name='table')

    def test__validate_data_missing_table(self, metadata, constraint):
        """Test ``_validate_data`` errors if the table is not in the data."""
        # Setup
        expected_error = re.escape("The table 'table' is missing from the data.")

        # Run & Assert
        with pytest.raises(ConstraintNotApplicableError, match=expected_error):
            constraint._validate_data({'OtherTable': pd.DataFrame()}, metadata)

    def test__validate_data_missing_columns(self, data, metadata, constraint):
        """Test ``_validate_data`` errors if a referenced column is not in the table."""
        # Setup
        del data['table']['a']
        expected_error = re.escape("The column(s) 'a' are missing from the table 'table'.")

        # Run & Assert
        with pytest.raises(ConstraintNotApplicableError, match=expected_error):
            constraint._validate_data(data, metadata)

    def test__fit(self, data, metadata, constraint):
        """Test ``_fit`` learns the combinations that are present in the real data."""
        # Run
        constraint._fit(data, metadata)

        # Assert
        pd.testing.assert_frame_equal(constraint._combinations, data['table'])

    def test__fit_drops_duplicates(self, data, metadata, constraint):
        """Test ``_fit`` only keeps one row per combination."""
        # Setup
        data['table'] = pd.concat([data['table']] * 2, ignore_index=True)

        # Run
        constraint._fit(data, metadata)

        # Assert
        expected = data['table'].iloc[:3]
        pd.testing.assert_frame_equal(constraint._combinations, expected)

    def test_fit_validates_the_data(self, data, metadata, constraint):
        """Test ``fit`` validates the real data before learning from it."""
        # Setup
        del data['table']['a']
        expected_error = re.escape("The column(s) 'a' are missing from the table 'table'.")

        # Run & Assert
        with pytest.raises(ConstraintNotApplicableError, match=expected_error):
            constraint.fit(data, metadata)

    def test__is_valid(self, data, metadata, constraint):
        """Test ``_is_valid`` method."""
        # Setup
        constraint.fit(data, metadata)

        # Run
        is_valid = constraint._is_valid(data, metadata)

        # Assert
        expected = pd.Series([True, True, True], name='a#b#c')
        pd.testing.assert_series_equal(is_valid['table'], expected)

    def test__is_valid_other_tables(self, data, metadata, constraint):
        """Test ``_is_valid`` considers every row of the other tables valid."""
        # Setup
        data['other_table'] = pd.DataFrame({'a': ['a', 'b']})
        constraint.fit(data, metadata)

        # Run
        is_valid = constraint._is_valid(data, metadata)

        # Assert
        expected = pd.Series([True, True])
        pd.testing.assert_series_equal(is_valid['other_table'], expected)

    def test__is_valid_with_invalid_values(self, data, metadata, constraint):
        """Test ``_is_valid`` flags the combinations that were not in the real data."""
        # Setup
        synthetic_data = {
            'table': pd.DataFrame({
                'a': ['a', 'b', 'c'],
                'b': ['d', 'E', 'f'],
                'c': ['g', 'h', 'I'],
            })
        }

        # Run
        constraint.fit(data, metadata)
        is_valid = constraint._is_valid(synthetic_data, metadata)

        # Assert
        expected = pd.Series([True, False, False], name='a#b#c')
        pd.testing.assert_series_equal(is_valid['table'], expected)

    def test__is_valid_non_string(self, metadata):
        """Test the ``_is_valid`` with non-string input."""
        # Setup
        data = {
            'table': pd.DataFrame({
                'b': [1, 2, 3],
                'c': ['g', 'h', 'i'],
                'd': [2.4, 1.23, 5.6],
            })
        }
        invalid_data = {
            'table': pd.DataFrame({
                'b': [6, 7, 8],
                'c': ['g', 'h', 'i'],
                'd': [2.4, 1.23, 5.6],
            })
        }
        instance = FixedCombinations(column_names=['b', 'c', 'd'], table_name='table')
        instance.fit(data, metadata)

        # Run
        valid_out = instance._is_valid(data, metadata)
        invalid_out = instance._is_valid(invalid_data, metadata)

        # Assert
        expected_valid_out = pd.Series([True, True, True], name='b#c#d')
        pd.testing.assert_series_equal(valid_out['table'], expected_valid_out)
        pd.testing.assert_series_equal(invalid_out['table'], ~expected_valid_out)

    def test__is_valid_with_nans(self, metadata):
        """Test the ``_is_valid`` method with missing values."""
        # Setup
        data = {
            'table': pd.DataFrame({
                'b': ['d', 'e', 'f', None, np.nan, 'f'],
                'c': ['g', 'h', None, None, None, None],
            })
        }
        invalid_data = {
            'table': pd.DataFrame({
                'b': ['D', np.nan, 'F'],
                'c': ['g', 'h', 'i'],
            })
        }
        instance = FixedCombinations(column_names=['b', 'c'], table_name='table')
        instance.fit(data, metadata)

        # Run
        valid_out = instance._is_valid(data, metadata)
        invalid_out = instance._is_valid(invalid_data, metadata)

        # Assert
        expected_valid_out = pd.Series([True] * 6, name='b#c')
        pd.testing.assert_series_equal(valid_out['table'], expected_valid_out)

        expected_invalid_out = pd.Series([False] * 3, name='b#c')
        pd.testing.assert_series_equal(invalid_out['table'], expected_invalid_out)

    def test_get_score(self, data, metadata, constraint):
        """Test get_score returns the proportion of valid rows."""
        # Setup
        constraint.fit(data, metadata)

        # Run & Assert
        assert constraint.get_score(data, metadata) == 1.0

    def test_get_score_not_fitted(self, data, metadata, constraint):
        """Test ``get_score`` errors if the constraint was not fitted first."""
        # Setup
        expected_error = re.escape(
            'FixedCombinations constraint must be called with ``fit`` first.'
        )

        # Run & Assert
        with pytest.raises(ConstraintNotApplicableError, match=expected_error):
            constraint.get_score(data, metadata)

    def test_get_score_invalid_synthetic_data(self, data, metadata, constraint):
        """Test ``get_score`` scores the synthetic data against the fitted combinations."""
        # Setup
        synthetic_data = {
            'table': pd.DataFrame({
                'a': ['a', 'b', 'c', 'a'],
                'b': ['d', 'E', 'f', 'd'],
                'c': ['g', 'h', 'I', 'g'],
            })
        }

        constraint.fit(data, metadata)

        # Run & Assert
        assert constraint.get_score(synthetic_data, metadata) == 0.5

    def test_get_score_empty_table(self, data, metadata, constraint):
        """Test ``get_score`` returns NaN when there are no rows to check."""
        # Setup
        data['table'] = data['table'].iloc[:0]
        constraint.fit(data, metadata)

        # Run & Assert
        assert pd.isna(constraint.get_score(data, metadata))
