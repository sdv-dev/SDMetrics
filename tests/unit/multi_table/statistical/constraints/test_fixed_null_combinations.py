import re

import numpy as np
import pandas as pd
import pytest

from sdmetrics.multi_table.statistical.constraints import FixedNullCombinations
from sdmetrics.multi_table.statistical.constraints.error import ConstraintNotApplicableError


@pytest.fixture
def data():
    return {
        'table': pd.DataFrame({
            'a': ['x', 'y', None, 'x'],
            'b': [1.0, np.nan, 3.0, 4.0],
        })
    }


@pytest.fixture
def metadata():
    return {
        'tables': {
            'table': {
                'columns': {
                    'a': {'sdtype': 'categorical'},
                    'b': {'sdtype': 'numerical'},
                }
            }
        }
    }


@pytest.fixture
def constraint():
    return FixedNullCombinations(table_name='table', column_names=['a', 'b'])


class TestFixedNullCombinations:
    def test___init__(self):
        """Test the ``__init__`` method sets the parameters."""
        # Run
        instance = FixedNullCombinations(column_names=['a', 'b'], table_name='table')

        # Assert
        assert instance.column_names == ['a', 'b']
        assert instance.table_name == 'table'
        assert instance.fix_category_values is True

    def test___init__invalid_parameters(self):
        """Test the ``__init__`` method errors with invalid arguments."""
        # Run and Assert
        err_msg = "The 'table_name' parameter must be a string."
        with pytest.raises(ValueError, match=err_msg):
            FixedNullCombinations(column_names=['a', 'b'], table_name=1)

        err_msg = re.escape("The 'column_names' parameter must be a list of strings.")
        with pytest.raises(ValueError, match=err_msg):
            FixedNullCombinations(column_names='a', table_name='table')

        with pytest.raises(ValueError, match=err_msg):
            FixedNullCombinations(column_names=['a', 2], table_name='table')

        err_msg = re.escape('FixedNullCombinations constraint requires at least two columns.')
        with pytest.raises(ValueError, match=err_msg):
            FixedNullCombinations(column_names=['a'], table_name='table')

        err_msg = re.escape('`fix_category_values` must be a boolean.')
        with pytest.raises(ValueError, match=err_msg):
            FixedNullCombinations(
                column_names=['a', 'b'], table_name='table', fix_category_values='yes'
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
        del data['table']['a']
        expected_error = re.escape("The column(s) 'a' are missing from the table 'table'.")

        # Run & Assert
        with pytest.raises(ConstraintNotApplicableError, match=expected_error):
            constraint._validate_data(data)

    def test__fit(self, data, metadata, constraint):
        """Test ``_fit`` learns the null and the value combinations of the real data."""
        # Run
        constraint._fit(data, metadata)

        # Assert
        expected_nan_combinations = frozenset([
            frozenset(['None']),
            frozenset(['a']),
            frozenset(['b']),
        ])
        assert constraint._nan_combinations == expected_nan_combinations
        assert constraint._categorical_columns == ['a']
        assert constraint._category_combinations_by_nanset == {
            frozenset(['None']): {('x',)},
            frozenset(['b']): {('y',)},
            frozenset(['a']): {(None,)},
        }

    def test__fit_without_nans(self, data, metadata, constraint):
        """Test ``_fit`` when the real data has no missing values."""
        # Setup
        data['table'] = pd.DataFrame({'a': ['x', 'y'], 'b': [1.0, 2.0]})

        # Run
        constraint._fit(data, metadata)

        # Assert
        assert constraint._nan_combinations == frozenset([frozenset(['None'])])
        assert constraint._category_combinations_by_nanset == {
            frozenset(['None']): {('x',), ('y',)}
        }

    def test__fit_boolean_columns(self, data, metadata, constraint):
        """Test ``_fit`` also fixes the values of the boolean columns."""
        # Setup
        metadata['tables']['table']['columns']['b']['sdtype'] = 'boolean'
        data['table']['b'] = [True, np.nan, False, True]

        # Run
        constraint._fit(data, metadata)

        # Assert
        assert constraint._categorical_columns == ['a', 'b']

    def test__fit_without_category_values(self, data, metadata):
        """Test ``_fit`` only learns the null combinations if ``fix_category_values`` is False."""
        # Setup
        instance = FixedNullCombinations(
            column_names=['a', 'b'], table_name='table', fix_category_values=False
        )

        # Run
        instance._fit(data, metadata)

        # Assert
        assert instance._nan_combinations == frozenset([
            frozenset(['None']),
            frozenset(['a']),
            frozenset(['b']),
        ])
        assert instance._category_combinations_by_nanset == {}

    def test_fit_validates_the_data(self, data, metadata, constraint):
        """Test ``fit`` validates the real data before learning from it."""
        # Setup
        del data['table']['a']
        expected_error = re.escape("The column(s) 'a' are missing from the table 'table'.")

        # Run & Assert
        with pytest.raises(ConstraintNotApplicableError, match=expected_error):
            constraint.fit(data, metadata)

    def test__is_valid(self, data, metadata, constraint):
        """Test ``_is_valid`` considers every row of the real data valid."""
        # Setup
        constraint.fit(data, metadata)

        # Run
        is_valid = constraint._is_valid(data, metadata)

        # Assert
        expected = pd.Series([True, True, True, True])
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
        """Test ``_is_valid`` flags the null and value combinations not in the real data."""
        # Setup
        synthetic_data = {
            'table': pd.DataFrame({
                'a': ['x', 'y', 'y', None],
                'b': [1.0, 2.0, np.nan, np.nan],
            })
        }
        constraint.fit(data, metadata)

        # Run
        is_valid = constraint._is_valid(synthetic_data, metadata)

        # Assert
        expected = pd.Series([True, False, True, False])
        pd.testing.assert_series_equal(is_valid['table'], expected)

    def test__is_valid_without_category_values(self, data, metadata):
        """Test ``_is_valid`` only checks the nullness if ``fix_category_values`` is False."""
        # Setup
        synthetic_data = {
            'table': pd.DataFrame({
                'a': ['x', 'y', 'y', None],
                'b': [1.0, 2.0, np.nan, np.nan],
            })
        }
        instance = FixedNullCombinations(
            column_names=['a', 'b'], table_name='table', fix_category_values=False
        )
        instance.fit(data, metadata)

        # Run
        is_valid = instance._is_valid(synthetic_data, metadata)

        # Assert
        expected = pd.Series([True, True, True, False])
        pd.testing.assert_series_equal(is_valid['table'], expected)

    def test__is_valid_with_nan_in_fit_data(self, data, metadata):
        """Test that constraint validates that nan combinations were in fitted data."""
        # Setup
        data = {
            'table': pd.DataFrame({
                'colA': range(10),
                'colB': [0, np.nan, np.nan, np.nan, 1, 2, 3, 3, 3, np.nan],
                'colC': ['A', None, 'A', None, 'B', 'A', 'C', 'A', 'A', 'B'],
                'colD': [
                    '01 Jan 2018',
                    '02 Jan 2018',
                    '04 Jan 2018',
                    None,
                    '05 Jan 2018',
                    '02 Jan 2018',
                    '03 Jan 2018',
                    '04 Jan 2018',
                    '05 Jan 2018',
                    '06 Jan 2018',
                ],
                'colE': [np.nan] * 10,
                'colF': ['A', 'B'] * 5,
            })
        }

        metadata = {
            'tables': {
                'table': {
                    'columns': {
                        'colA': {'sdtype': 'numerical'},
                        'colB': {'sdtype': 'numerical'},
                        'colC': {'sdtype': 'categorical'},
                        'colD': {'sdtype': 'datetime'},
                        'colE': {'sdtype': 'numerical'},
                        'colF': {'sdtype': 'numerical'},
                    }
                }
            }
        }
        constraint = FixedNullCombinations(column_names=['colB', 'colC'], table_name='table')
        constraint.metadata = metadata
        constraint._fitted = True
        constraint._nan_combinations = frozenset({frozenset({'colB', 'colC'}), frozenset({'None'})})

        # Run
        is_valid_dict = constraint._is_valid(data)

        # Assert
        assert set(is_valid_dict.keys()) == {'table'}
        is_valid = is_valid_dict['table']
        expected_is_valid = pd.Series([
            True,
            True,
            False,
            True,
            True,
            True,
            True,
            True,
            True,
            False,
        ])
        pd.testing.assert_series_equal(is_valid, expected_is_valid)

    def test__is_valid_with_category_enforcement(self):
        """Test that categorical values are enforced conditional on nullness."""
        # Setup
        metadata = {
            'tables': {
                'table': {
                    'columns': {
                        'status': {'sdtype': 'categorical'},
                        'date': {'sdtype': 'datetime', 'datetime_format': '%Y-%m-%d'},
                    }
                }
            }
        }
        train = {
            'table': pd.DataFrame({
                'status': ['IN_PROGRESS', 'NOT_STARTED', 'RESOLVED_OK'],
                'date': [None, None, '2025-05-01'],
            })
        }
        constraint = FixedNullCombinations(column_names=['status', 'date'], table_name='table')
        constraint._fit(train, metadata)
        constraint.metadata = metadata
        constraint._fitted = True

        # Candidate data to validate
        to_validate = {
            'table': pd.DataFrame({
                # Valid: seen in training (null date)
                'status': ['IN_PROGRESS', 'RESOLVED_OK', 'IN_PROGRESS'],
                'date': [None, '2025-05-03', '2025-05-09'],
            })
        }

        # Run
        is_valid_dict = constraint._is_valid(to_validate)

        # Assert
        is_valid = is_valid_dict['table']
        pd.testing.assert_series_equal(
            is_valid.reset_index(drop=True), pd.Series([True, True, False])
        )

    def test__is_valid_not_fitted(self, data, metadata):
        """Test the constraint sees all combinations as valid before fitting."""
        # Setup
        constraint = FixedNullCombinations(column_names=['a', 'b'], table_name='table')
        constraint.metadata = metadata

        # Run
        is_valid_dict = constraint.is_valid(data, metadata)

        # Assert
        assert set(is_valid_dict.keys()) == {'table'}
        is_valid = is_valid_dict['table']
        expected_is_valid = pd.Series([True] * 4)
        pd.testing.assert_series_equal(is_valid, expected_is_valid)

    def test_get_score(self, data, metadata, constraint):
        """Test ``get_score`` returns the proportion of valid rows."""
        # Setup
        constraint.fit(data, metadata)

        # Run & Assert
        assert constraint.get_score(data, metadata) == 1.0

    def test_get_score_invalid_synthetic_data(self, data, metadata, constraint):
        """Test ``get_score`` scores the synthetic data against the fitted combinations."""
        # Setup
        synthetic_data = {
            'table': pd.DataFrame({
                'a': ['x', 'y', 'y', None],
                'b': [1.0, 2.0, np.nan, np.nan],
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
