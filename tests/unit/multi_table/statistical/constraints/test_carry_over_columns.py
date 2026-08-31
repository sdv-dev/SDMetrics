import re

import numpy as np
import pandas as pd
import pytest

from sdmetrics.multi_table.statistical.constraints import CarryOverColumns
from sdmetrics.multi_table.statistical.constraints.error import ConstraintNotApplicableError


@pytest.fixture
def metadata():
    return {
        'tables': {
            'main_table': {
                'columns': {
                    'primary_key': {'sdtype': 'id'},
                    'parent_1': {'sdtype': 'categorical'},
                },
                'primary_key': 'primary_key',
            },
            'carry_over_1': {
                'columns': {
                    'child_1': {'sdtype': 'categorical'},
                    'child_2': {'sdtype': 'categorical'},
                    'key_column_1': {'sdtype': 'id'},
                    'key_column_2': {'sdtype': 'id'},
                },
            },
            'carry_over_2': {
                'columns': {'child_3': {'sdtype': 'categorical'}, 'foreign_key': {'sdtype': 'id'}},
            },
        },
    }


@pytest.fixture
def data():
    return {
        'main_table': pd.DataFrame({
            'primary_key': [1, 2, 3, 4, 5, 6],
            'parent_1': ['a', 'b', 'c', 'a', 'b', 'c'],
        }),
        'carry_over_1': pd.DataFrame({
            'child_1': ['a', 'a', 'c', 'c', 'd', 'e', 'f'],
            'child_2': ['b', 'b', 'a', 'a', 'd', 'e', 'f'],
            'key_column_1': [1, 1, 3, 3, 7, 8, 9],
            'key_column_2': [2, 2, 4, 4, 7, 8, 9],
        }),
        'carry_over_2': pd.DataFrame({
            'child_3': ['a', 'b', 'c', 'b', 'c', 'd', 'e'],
            'foreign_key': [1, 2, 3, 5, 6, 7, 8],
        }),
    }


@pytest.fixture
def common_column_info():
    return [
        {
            'table_name': 'main_table',
            'key_column_name': 'primary_key',
            'carryover_column_name': 'parent_1',
        },
        {
            'table_name': 'carry_over_1',
            'key_column_name': 'key_column_1',
            'carryover_column_name': 'child_1',
        },
        {
            'table_name': 'carry_over_1',
            'key_column_name': 'key_column_2',
            'carryover_column_name': 'child_2',
        },
        {
            'table_name': 'carry_over_2',
            'key_column_name': 'foreign_key',
            'carryover_column_name': 'child_3',
        },
    ]


@pytest.fixture
def constraint(common_column_info):
    return CarryOverColumns(common_column_info=common_column_info)


class TestCarryOverColumns:
    def test___init__invalid_parameters(self, common_column_info):
        """Test the ``__init__`` method errors with invalid arguments."""
        # Setup
        err_msg = re.escape('`common_column_info` must be a list.')

        # Run and Assert
        with pytest.raises(TypeError, match=err_msg):
            CarryOverColumns(common_column_info=common_column_info[0])

    def test___init__invalid_keys(self, common_column_info):
        """Test the ``__init__`` method errors if an entry has the wrong keys."""
        # Setup
        err_msg = re.escape(
            "Each element of `common_column_info` must have the keys 'table_name', "
            "'carryover_column_name', and 'key_column_name'."
        )
        not_a_dict_msg = re.escape('Each element of `common_column_info` must be a dictionary.')

        # Run and Assert
        with pytest.raises(ValueError, match=err_msg):
            CarryOverColumns(common_column_info=[common_column_info[0], {'table_name': 'tableA'}])

        with pytest.raises(TypeError, match=not_a_dict_msg):
            CarryOverColumns(common_column_info=[common_column_info[0], 'tableA'])

    def test___init__invalid_values(self, common_column_info):
        """Test the ``__init__`` method errors if an entry has non string values."""
        # Setup
        invalid_info = {**common_column_info[1], 'key_column_name': 1}
        err_msg = re.escape(
            "The values of 'table_name', 'carryover_column_name', and 'key_column_name' "
            'in each element of `common_column_info` must be strings.'
        )

        # Run and Assert
        with pytest.raises(TypeError, match=err_msg):
            CarryOverColumns(common_column_info=[common_column_info[0], invalid_info])

    def test__validate_data_missing_table(self, data, metadata, constraint):
        """Test ``_validate_data`` errors if one of the tables is not in the data."""
        # Setup
        del data['main_table']
        expected_error = re.escape("The table 'main_table' is missing from the data.")

        # Run and Assert
        with pytest.raises(ConstraintNotApplicableError, match=expected_error):
            constraint._validate_data(data, metadata)

    def test__validate_data_missing_key_column(self, data, metadata, constraint):
        """Test ``_validate_data`` errors if the key column is not in the table."""
        # Setup
        del data['main_table']['primary_key']
        expected_error = re.escape(
            "The column(s) 'primary_key' are missing from the table 'main_table'."
        )

        # Run and Assert
        with pytest.raises(ConstraintNotApplicableError, match=expected_error):
            constraint._validate_data(data, metadata)

    def test__validate_data_missing_carryover_column(self, data, metadata, constraint):
        """Test ``_validate_data`` errors if the carryover column is not in the table."""
        # Setup
        del data['carry_over_1']['child_1']
        expected_error = re.escape(
            "The column(s) 'child_1' are missing from the table 'carry_over_1'."
        )

        # Run and Assert
        with pytest.raises(ConstraintNotApplicableError, match=expected_error):
            constraint._validate_data(data, metadata)

    def test__is_valid(self, data, metadata, constraint):
        """Test ``_is_valid`` considers every row valid when the values match up."""
        # Run
        is_valid = constraint._is_valid(data, metadata)

        # Assert
        pd.testing.assert_series_equal(is_valid['main_table'], pd.Series([True] * 6))
        pd.testing.assert_series_equal(is_valid['carry_over_1'], pd.Series([True] * 7))
        pd.testing.assert_series_equal(is_valid['carry_over_2'], pd.Series([True] * 7))

    def test__is_valid_with_inconsistent_carry_over_table(self, data, metadata, constraint):
        """Test ``_is_valid`` flags a key that is inconsistent within a single table."""
        # Setup
        data['carry_over_1'].loc[3, 'child_1'] = 'd'

        # Run
        is_valid = constraint._is_valid(data, metadata)

        # Assert
        expected_carry_over_1 = pd.Series([True, True, True, False, True, True, True])
        pd.testing.assert_series_equal(is_valid['main_table'], pd.Series([True] * 6))
        pd.testing.assert_series_equal(is_valid['carry_over_1'], expected_carry_over_1)
        pd.testing.assert_series_equal(is_valid['carry_over_2'], pd.Series([True] * 7))

    def test__is_valid_with_invalid_values(self, data, metadata, constraint):
        """Test ``_is_valid`` flags every row of a key that does not match up."""
        # Setup
        data['carry_over_2'].loc[0, 'child_3'] = 'b'
        data['carry_over_2'].loc[3, 'child_3'] = 'z'

        # Run
        is_valid = constraint._is_valid(data, metadata)

        # Assert
        expected_carry_over_2 = pd.Series([False, True, True, False, True, True, True])
        pd.testing.assert_series_equal(is_valid['main_table'], pd.Series([True] * 6))
        pd.testing.assert_series_equal(is_valid['carry_over_1'], pd.Series([True] * 7))
        pd.testing.assert_series_equal(is_valid['carry_over_2'], expected_carry_over_2)

    def test__is_valid_with_nans(self, data, metadata, constraint):
        """Test ``_is_valid`` treats a missing carryover value as its own value."""
        # Setup
        data['carry_over_1'].loc[4, 'child_1'] = np.nan

        # Run
        is_valid = constraint._is_valid(data, metadata)

        # Assert
        pd.testing.assert_series_equal(is_valid['main_table'], pd.Series([True] * 6))
        pd.testing.assert_series_equal(is_valid['carry_over_1'], pd.Series([True] * 7))
        pd.testing.assert_series_equal(is_valid['carry_over_2'], pd.Series([True] * 7))

    def test__is_valid_with_unmatched_key(self, data, metadata, constraint):
        """Test ``_is_valid`` considers a key that is only in one table valid."""
        # Setup
        data['carry_over_2'].loc[6, 'foreign_key'] = 99

        # Run
        is_valid = constraint._is_valid(data, metadata)

        # Assert
        pd.testing.assert_series_equal(is_valid['main_table'], pd.Series([True] * 6))
        pd.testing.assert_series_equal(is_valid['carry_over_1'], pd.Series([True] * 7))
        pd.testing.assert_series_equal(is_valid['carry_over_2'], pd.Series([True] * 7))

    def test__is_valid_other_tables(self, data, metadata, constraint):
        """Test ``_is_valid`` considers every row of the other tables valid."""
        # Setup
        data['other_table'] = pd.DataFrame({'a': ['a', 'b']})

        # Run
        is_valid = constraint._is_valid(data, metadata)

        # Assert
        pd.testing.assert_series_equal(is_valid['other_table'], pd.Series([True, True]))

    def test__is_valid_mismatch(self, common_column_info):
        """Test that rows are marked invalid if there are mismatching carry over columns.

        This test will check rows where:
            - The carry over column is mismatched for the same key value.
            - The same key value has multiple carry over column values in one table.
        """
        # Setup
        data = {
            'main_table': pd.DataFrame({
                'primary_key': [1, 2, 3, 4, 5, 6],
                'parent_1': ['a', 'b', 'c', 'a', 'b', 'c'],
            }),
            'carry_over_1': pd.DataFrame({
                'child_1': ['a', 'b', 'c', 'c', 'd', 'e', 'f'],
                'child_2': ['b', 'b', 'a', 'a', 'd', 'e', 'f'],
                'key_column_1': [1, 1, 3, 3, 7, 8, 9],
                'key_column_2': [2, 2, 4, 4, 7, 8, 9],
            }),
            'carry_over_2': pd.DataFrame({
                'child_3': ['d', 'b', 'c', 'b', 'c', 'd', 'e'],
                'foreign_key': [1, 2, 3, 5, 6, 7, 8],
            }),
        }
        common_column_info = [
            {
                'table_name': 'main_table',
                'key_column_name': 'primary_key',
                'carryover_column_name': 'parent_1',
            },
            {
                'table_name': 'carry_over_1',
                'key_column_name': 'key_column_1',
                'carryover_column_name': 'child_1',
            },
            {
                'table_name': 'carry_over_2',
                'key_column_name': 'foreign_key',
                'carryover_column_name': 'child_3',
            },
        ]
        constraint = CarryOverColumns(common_column_info)

        # Run
        valid_rows = constraint._is_valid(data)

        # Assert
        expected = {
            'main_table': pd.Series([True] * 6),
            'carry_over_1': pd.Series([True, False, True, True, True, True, True]),
            'carry_over_2': pd.Series([False, True, True, True, True, True, True]),
        }
        for key in data.keys():
            pd.testing.assert_series_equal(expected[key], valid_rows[key])

    def test_get_score(self, data, metadata, constraint):
        """Test ``get_score`` returns the proportion of valid rows."""
        # Run and Assert
        assert constraint.get_score(data, metadata) == 1.0

    def test_get_score_invalid_data(self, data, metadata, constraint):
        """Test ``get_score`` counts the rows of every table involved."""
        # Setup
        data['carry_over_1'].loc[1, 'child_1'] = 'b'

        # Run and Assert
        assert constraint.get_score(data, metadata) == 0.95

    def test_get_score_empty_tables(self, data, metadata, constraint):
        """Test ``get_score`` returns NaN when there are no rows to check."""
        # Setup
        data = {table: table_data.iloc[:0] for table, table_data in data.items()}

        # Run and Assert
        assert pd.isna(constraint.get_score(data, metadata))
