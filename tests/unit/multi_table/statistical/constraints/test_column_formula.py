import re

import numpy as np
import pandas as pd
import pytest

from sdmetrics.multi_table.statistical.constraints import ColumnFormula
from sdmetrics.multi_table.statistical.constraints.error import ConstraintNotApplicableError

MODULE = __name__
NOT_A_FUNCTION = 1


def calculate_total(data):
    return data['subtotal'] + data['tax']


def calculate_partial_total(data):
    """Return a value only for the rows where the subtotal is not too large."""
    output = data['subtotal'] + data['tax']
    return output[data['subtotal'] <= 30]


def calculate_list(data):
    return [1] * len(data)


def crash_formula(data):
    raise Exception('This formula is broken.')


def _get_constraint(formula_function_name):
    return ColumnFormula(
        input_column_names=['subtotal', 'tax'],
        output_column_name='total',
        formula_function_name=formula_function_name,
        module_name=__name__,
        table_name='table',
    )


@pytest.fixture
def data():
    return {
        'table': pd.DataFrame({
            'subtotal': [10, 20, 30, 40],
            'tax': [1, 2, 3, 4],
            'total': [11, 22, 33, 44],
            'other': ['a', 'b', 'c', 'd'],
        })
    }


@pytest.fixture
def metadata():
    return {
        'tables': {
            'table': {
                'columns': {
                    'subtotal': {'sdtype': 'numerical'},
                    'tax': {'sdtype': 'numerical'},
                    'total': {'sdtype': 'numerical'},
                    'other': {'sdtype': 'categorical'},
                }
            }
        }
    }


@pytest.fixture
def constraint():
    return _get_constraint('calculate_total')


class TestColumnFormula:
    def test___init__(self):
        """Test the ``__init__`` method sets the parameters and loads the formula function."""
        # Run
        instance = _get_constraint('calculate_total')

        # Assert
        assert instance.input_column_names == ['subtotal', 'tax']
        assert instance.output_column_name == 'total'
        assert instance.formula_function_name == 'calculate_total'
        assert instance.module_name == __name__
        assert instance.table_name == 'table'
        assert instance._formula_function is calculate_total

    def test___init__default_module_name(self):
        """Test the ``__init__`` method looks for the function in ``__main__`` by default."""
        # Setup
        expected_error = re.escape(
            "Module '__main__' does not contain a function named 'calculate_total'."
        )

        # Run and Assert
        with pytest.raises(ValueError, match=expected_error):
            ColumnFormula(
                input_column_names=['a', 'b'],
                output_column_name='c',
                formula_function_name='calculate_total',
                table_name='table',
            )

    def test___init__unknown_module(self):
        """Test the ``__init__`` method errors if the module cannot be imported."""
        # Setup
        expected_error = re.escape("Unable to import module 'not_a_real_module'.")

        # Run and Assert
        with pytest.raises(ValueError, match=expected_error):
            ColumnFormula(
                input_column_names=['a', 'b'],
                output_column_name='c',
                formula_function_name='calculate_total',
                module_name='not_a_real_module',
                table_name='table',
            )

    def test___init__unknown_function(self):
        """Test the ``__init__`` method errors if the function is not in the module."""
        # Setup
        expected_error = re.escape(
            f"Module '{__name__}' does not contain a function named 'not_a_real_function'."
        )

        # Run and Assert
        with pytest.raises(ValueError, match=expected_error):
            _get_constraint('not_a_real_function')

    def test___init__not_a_function(self):
        """Test the ``__init__`` method errors if the module attribute is not callable."""
        # Setup
        expected_error = re.escape(
            '`formula_function_name` must reference a callable function. '
            "'NOT_A_FUNCTION' is not callable."
        )

        # Run and Assert
        with pytest.raises(TypeError, match=expected_error):
            _get_constraint('NOT_A_FUNCTION')

    def test___init__invalid_parameters(self):
        """Test the ``__init__`` method errors with invalid arguments."""
        # Setup
        parameters = {
            'input_column_names': ['a', 'b'],
            'output_column_name': 'c',
            'formula_function_name': 'my_formula',
            'table_name': 'table',
        }

        # Run and Assert
        err_msg = "The 'table_name' parameter must be a string."
        with pytest.raises(ValueError, match=err_msg):
            ColumnFormula(**{**parameters, 'table_name': 1})

        err_msg = re.escape(
            "The 'input_column_names' parameter must be a non-empty list of strings."
        )
        with pytest.raises(ValueError, match=err_msg):
            ColumnFormula(**{**parameters, 'input_column_names': 'a'})

        with pytest.raises(ValueError, match=err_msg):
            ColumnFormula(**{**parameters, 'input_column_names': []})

        err_msg = re.escape("The 'output_column_name' parameter must be a string.")
        with pytest.raises(ValueError, match=err_msg):
            ColumnFormula(**{**parameters, 'output_column_name': ['c']})

        err_msg = re.escape("The 'formula_function_name' parameter must be a string.")
        with pytest.raises(ValueError, match=err_msg):
            ColumnFormula(**{**parameters, 'formula_function_name': calculate_total})

        err_msg = re.escape("The 'module_name' parameter must be a string.")
        with pytest.raises(ValueError, match=err_msg):
            ColumnFormula(**{**parameters, 'module_name': 1})

    def test__validate_data_missing_table(self, metadata, constraint):
        """Test ``_validate_data`` errors if the table is not in the data."""
        # Setup
        expected_error = re.escape("The table 'table' is missing from the data.")

        # Run and Assert
        with pytest.raises(ConstraintNotApplicableError, match=expected_error):
            constraint._validate_data({'OtherTable': pd.DataFrame()}, metadata)

    def test__validate_data_missing_input_column(self, data, metadata, constraint):
        """Test ``_validate_data`` errors if an input column is not in the table."""
        # Setup
        del data['table']['tax']
        expected_error = re.escape("The column(s) 'tax' are missing from the table 'table'.")

        # Run and Assert
        with pytest.raises(ConstraintNotApplicableError, match=expected_error):
            constraint._validate_data(data, metadata)

    def test__validate_data_missing_output_column(self, data, metadata, constraint):
        """Test ``_validate_data`` errors if the output column is not in the table."""
        # Setup
        del data['table']['total']
        expected_error = re.escape("The column(s) 'total' are missing from the table 'table'.")

        # Run and Assert
        with pytest.raises(ConstraintNotApplicableError, match=expected_error):
            constraint._validate_data(data, metadata)

    def test__get_formula_output_missing_input_columns(self, data, constraint):
        """Test ``_get_formula_output`` errors if an input column is missing."""
        # Setup
        del data['table']['subtotal']
        expected_error = re.escape("Data is missing input columns ['subtotal'] for the formula.")

        # Run and Assert
        with pytest.raises(ConstraintNotApplicableError, match=expected_error):
            constraint._get_formula_output(data['table'])

    def test__get_formula_output_not_a_series(self, data):
        """Test ``_get_formula_output`` errors if the function does not return a Series."""
        # Setup
        instance = _get_constraint('calculate_list')
        expected_error = re.escape('The formula function must return a pandas Series.')

        # Run and Assert
        with pytest.raises(TypeError, match=expected_error):
            instance._get_formula_output(data['table'])

    def test__get_formula_output_raises_an_error(self, data):
        """Test ``_get_formula_output`` lets the error of the formula function through."""
        # Setup
        instance = _get_constraint('crash_formula')

        # Run and Assert
        with pytest.raises(Exception, match='This formula is broken.'):
            instance._get_formula_output(data['table'])

    def test__is_valid(self, data, metadata, constraint):
        """Test ``_is_valid`` considers every row that matches the formula valid."""
        # Setup
        constraint._validate_data(data, metadata)

        # Run
        is_valid = constraint._is_valid(data, metadata)

        # Assert
        expected = pd.Series([True, True, True, True])
        pd.testing.assert_series_equal(is_valid['table'], expected)

    def test__is_valid_with_invalid_values(self, data, metadata, constraint):
        """Test ``_is_valid`` flags the rows that do not match the formula."""
        # Setup
        data['table']['total'] = [11, 999, 33, 550]
        constraint._validate_data(data, metadata)

        # Run
        is_valid = constraint._is_valid(data, metadata)

        # Assert
        expected = pd.Series([True, False, True, False])
        pd.testing.assert_series_equal(is_valid['table'], expected)

    def test__is_valid_with_nans(self, data, metadata, constraint):
        """Test ``_is_valid`` only accepts a null output when the formula is also null."""
        # Setup
        data['table']['subtotal'] = [10, np.nan, 30, 40]
        data['table']['total'] = [11, np.nan, np.nan, 44]
        constraint._validate_data(data, metadata)

        # Run
        is_valid = constraint._is_valid(data, metadata)

        # Assert
        expected = pd.Series([True, True, False, True])
        pd.testing.assert_series_equal(is_valid['table'], expected)

    def test__is_valid_with_missing_formula_output(self, data, metadata):
        """Test ``_is_valid`` flags the rows that the formula did not return a value for."""
        # Setup
        instance = _get_constraint('calculate_partial_total')

        # Run
        is_valid = instance._is_valid(data, metadata)

        # Assert
        expected = pd.Series([True, True, True, False])
        pd.testing.assert_series_equal(is_valid['table'], expected)

    def test__is_valid_with_other_table(self, data, metadata):
        """Test that ``_is_valid`` returns rows that match the formula."""
        # Setup
        data['other_table'] = pd.DataFrame({'id': [0, 1]})
        instance = _get_constraint('calculate_total')
        data['table'].loc[1, 'total'] = 23

        # Run
        is_valid = instance._is_valid(data, metadata)

        # Assert
        assert set(is_valid) == {'table', 'other_table'}
        pd.testing.assert_series_equal(is_valid['table'], pd.Series([True, False, True, True]))
        pd.testing.assert_series_equal(is_valid['other_table'], pd.Series([True, True]))

    def test__is_valid_output_column_missing(self, data, metadata):
        """Test that is_valid errors when the output column is missing."""
        # Setup
        data = {'table': data['table'].drop(columns=['total'])}
        instance = _get_constraint('calculate_total')

        # Run and Assert
        expected_msg = re.escape("Data is missing output column 'total' for the formula.")
        with pytest.raises(ConstraintNotApplicableError, match=expected_msg):
            instance._is_valid(data, metadata)

    def test_get_score(self, data, metadata, constraint):
        """Test ``get_score`` returns the proportion of valid rows."""
        # Run and Assert
        assert constraint.get_score(data, metadata) == 1.0

    def test_get_score_invalid_data(self, data, metadata, constraint):
        """Test ``get_score`` returns the proportion of rows that match the formula."""
        # Setup
        data['table']['total'] = [11, 999, 33, 550]

        # Run and Assert
        assert constraint.get_score(data, metadata) == 0.5

    def test_get_score_empty_table(self, data, metadata, constraint):
        """Test ``get_score`` returns NaN when there are no rows to check."""
        # Setup
        data['table'] = data['table'].iloc[:0]

        # Run and Assert
        assert pd.isna(constraint.get_score(data, metadata))
