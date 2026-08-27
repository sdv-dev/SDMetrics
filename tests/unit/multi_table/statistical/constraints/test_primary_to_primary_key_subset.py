import re
from unittest.mock import Mock

import pandas as pd
import pytest

from sdmetrics.multi_table.statistical.constraints import PrimaryToPrimaryKeySubset
from sdmetrics.multi_table.statistical.constraints.error import ConstraintNotApplicableError


@pytest.fixture
def data():
    return {
        'main_table': pd.DataFrame({
            'primary_key': [1, 2, 3, 4, 5],
            'condition_column': [
                'conditional_value_1',
                'conditional_value_2',
                'conditional_value_1',
                'conditional_value_2',
                'conditional_value_1',
            ],
        }),
        'table_1': pd.DataFrame({
            'col_1': [1, 3, 5],
            'col_2': [7, 8, 10],
            'col_3': ['A', 'A', 'B'],
        }),
        'table_2': pd.DataFrame({'col_4': ['2', '4'], 'col_5': [14.5, 16.7], 'col_6': ['D', 'E']}),
    }


@pytest.fixture
def metadata():
    return {
        'tables': {
            'main_table': {
                'columns': {
                    'primary_key': {'sdtype': 'id'},
                    'condition_column': {'sdtype': 'categorical'},
                },
                'primary_key': 'primary_key',
            },
            'table_1': {
                'columns': {
                    'col_1': {'sdtype': 'id'},
                    'col_2': {'sdtype': 'numerical'},
                    'col_3': {'sdtype': 'categorical'},
                },
                'primary_key': 'col_1',
            },
            'table_2': {
                'columns': {
                    'col_4': {'sdtype': 'id'},
                    'col_5': {'sdtype': 'numerical'},
                    'col_6': {'sdtype': 'categorical'},
                },
                'primary_key': 'col_4',
            },
        },
        'relationships': [{'table_1': ['conditional_value_1'], 'table_2': ['conditional_value_2']}],
    }


@pytest.fixture
def constraint():
    return PrimaryToPrimaryKeySubset(
        main_table_name='main_table',
        conditional_column_name='condition_column',
        relationships={'table_1': ['conditional_value_1'], 'table_2': ['conditional_value_2']},
    )


class TestPrimaryToPrimaryKeySubset:
    def test___init__(self, constraint):
        """Test the ``__init__`` method sets the parameters."""
        # Assert
        assert constraint.main_table_name == 'main_table'
        assert constraint.conditional_column_name == 'condition_column'
        assert constraint.relationships == {
            'table_1': ['conditional_value_1'],
            'table_2': ['conditional_value_2'],
        }

    def test___init__invalid_parameters(self):
        """Test the ``__init__`` method errors with invalid arguments."""
        # Setup
        parameters = {
            'main_table_name': 'main_table',
            'conditional_column_name': 'condition_column',
            'relationships': {
                'table_1': ['conditional_value_1'],
                'table_2': ['conditional_value_2'],
            },
        }
        err_msg = re.escape('`main_table_name` and `conditional_column_name` must be strings')

        # Run and Assert
        with pytest.raises(ValueError, match=err_msg):
            PrimaryToPrimaryKeySubset(**{**parameters, 'main_table_name': 1})

        with pytest.raises(ValueError, match=err_msg):
            PrimaryToPrimaryKeySubset(**{**parameters, 'conditional_column_name': 1})

    def test___init__invalid_relationships(self):
        """Test the ``__init__`` method errors if a relationship is malformed."""
        # Setup
        parameters = {
            'main_table_name': 'main_table',
            'conditional_column_name': 'condition_column',
            'relationships': {
                'table_1': ['conditional_value_1'],
                'table_2': ['conditional_value_2'],
            },
        }
        err_msg = re.escape(
            '`relationships` must be a a dict that maps the name of the connected table to a '
            'list of values that are acceptable for a connection to be made.'
        )

        # Run and Assert
        with pytest.raises(ValueError, match=err_msg):
            PrimaryToPrimaryKeySubset(**{**parameters, 'relationships': ['table_1']})

        with pytest.raises(ValueError, match=err_msg):
            PrimaryToPrimaryKeySubset(**{
                **parameters,
                'relationships': {'table_1': 'conditional_value_1'},
            })

        with pytest.raises(ValueError, match=err_msg):
            PrimaryToPrimaryKeySubset(**{**parameters, 'relationships': {1: ['a']}})

    def test__validate_data_missing_table(self, data, metadata, constraint):
        """Test ``_validate_data`` errors if a connected table is not in the data."""
        # Setup
        del data['table_2']
        expected_error = re.escape("The table 'table_2' is missing from the data.")

        # Run & Assert
        with pytest.raises(ConstraintNotApplicableError, match=expected_error):
            constraint._validate_data(data, metadata)

    def test__validate_data_missing_main_table(self, data, metadata, constraint):
        """Test ``_validate_data`` errors if the main table is not in the data."""
        # Setup
        del data['main_table']
        expected_error = re.escape("The table 'main_table' is missing from the data.")

        # Run & Assert
        with pytest.raises(ConstraintNotApplicableError, match=expected_error):
            constraint._validate_data(data, metadata)

    def test__validate_data_missing_column(self, data, metadata, constraint):
        """Test ``_validate_data`` errors if the conditional column is not in the main table."""
        # Setup
        del data['main_table']['condition_column']
        expected_error = re.escape(
            "The column(s) 'condition_column' are missing from the table 'main_table'."
        )

        # Run & Assert
        with pytest.raises(ConstraintNotApplicableError, match=expected_error):
            constraint._validate_data(data, metadata)

    def test__validate_data_missing_connected_primary_key_column(self, data, metadata, constraint):
        """Test ``_validate_data`` errors if a connected table does not hold its primary key."""
        # Setup
        del data['table_1']['col_1']
        expected_error = re.escape("The column(s) 'col_1' are missing from the table 'table_1'.")

        # Run & Assert
        with pytest.raises(ConstraintNotApplicableError, match=expected_error):
            constraint._validate_data(data, metadata)

    def test__validate_data_missing_primary_key(self, data, metadata, constraint):
        """Test ``_validate_data`` errors if a table has no primary key in the metadata."""
        # Setup
        del metadata['tables']['table_1']['primary_key']
        expected_error = re.escape(
            "The table 'table_1' does not have a primary key in the metadata."
        )

        # Run & Assert
        with pytest.raises(ConstraintNotApplicableError, match=expected_error):
            constraint._validate_data(data, metadata)

    def test__get_metadata_parameters(self, metadata):
        """Test the ``_get_metadata_parameters`` method."""
        # Setup
        relationships = {'table_1': ['conditional_value_1'], 'table_2': ['conditional_value_2']}
        constraint = PrimaryToPrimaryKeySubset('main_table', 'condition_column', relationships)

        # Run
        results = constraint._get_metadata_parameters(metadata)

        # Assert
        assert results[0] == {'main_table': 'primary_key', 'table_1': 'col_1', 'table_2': 'col_4'}
        assert results[1] == ['primary_key', 'condition_column']
        assert results[2] == {
            'table_1': {'col_2': 'table_1_col_2', 'col_3': 'table_1_col_3'},
            'table_2': {'col_5': 'table_2_col_5', 'col_6': 'table_2_col_6'},
        }

    def test__is_valid(self):
        """Test that rows with mismatched keys are marked invalid."""
        # Setup
        instance = PrimaryToPrimaryKeySubset(
            'main', 'color', relationships={'ref1': ['blue'], 'ref2': ['green', 'red']}
        )
        instance._get_metadata_parameters = Mock(
            return_value=(
                {
                    'main': 'pk1',
                    'ref1': 'pk2',
                    'ref2': 'pk3',
                },
                '',
                '',
            )
        )
        data = {
            'main': pd.DataFrame({
                'pk1': range(20),
                'color': (['blue'] * 5) + (['green'] * 5) + (['red'] * 5) + (['orange'] * 5),
            }),
            'ref1': pd.DataFrame({'pk2': [9, 1, 4, 2, 6, 21]}),
            'ref2': pd.DataFrame({'pk3': [6, 7, 8, 1, 16, 22]}),
        }

        # Run
        valid_rows = instance._is_valid(data)

        # Assert
        expected = {
            'main': pd.Series([True] * 20),
            'ref1': pd.Series([False, True, True, True, False, False]),
            'ref2': pd.Series([True, True, True, False, False, False]),
        }
        for table, data in expected.items():
            pd.testing.assert_series_equal(data, valid_rows[table])

    def test__is_valid_default(self, data, metadata):
        """Test ``_is_valid`` considers valid every connected row that is allowed."""
        # Setup
        instance = PrimaryToPrimaryKeySubset(
            main_table_name='main_table',
            conditional_column_name='condition_column',
            relationships={'table_1': ['conditional_value_1']},
        )

        # Run
        is_valid = instance._is_valid(data, metadata)

        # Assert
        pd.testing.assert_series_equal(is_valid['main_table'], pd.Series([True] * 5))
        pd.testing.assert_series_equal(is_valid['table_1'], pd.Series([True] * 3))

    def test__is_valid_with_invalid_values(self, data, metadata):
        """Test ``_is_valid`` flags a connected row whose main row is not allowed."""
        # Setup
        data['table_1']['col_1'] = [1, 2, 5]
        instance = PrimaryToPrimaryKeySubset(
            main_table_name='main_table',
            conditional_column_name='condition_column',
            relationships={'table_1': ['conditional_value_1']},
        )

        # Run
        is_valid = instance._is_valid(data, metadata)

        # Assert
        pd.testing.assert_series_equal(is_valid['table_1'], pd.Series([True, False, True]))

    def test__is_valid_with_unknown_key(self, data, metadata):
        """Test ``_is_valid`` flags a connected row that the main table does not hold."""
        # Setup
        data['table_1']['col_1'] = [1, 3, 99]
        instance = PrimaryToPrimaryKeySubset(
            main_table_name='main_table',
            conditional_column_name='condition_column',
            relationships={'table_1': ['conditional_value_1']},
        )

        # Run
        is_valid = instance._is_valid(data, metadata)

        # Assert
        pd.testing.assert_series_equal(is_valid['table_1'], pd.Series([True, True, False]))

    def test__is_valid_several_relationships(self, data, metadata, constraint):
        """Test ``_is_valid`` checks every connected table against its own values."""
        # Setup
        data['table_2']['col_4'] = [2, 4]

        # Run
        is_valid = constraint._is_valid(data, metadata)

        # Assert
        pd.testing.assert_series_equal(is_valid['main_table'], pd.Series([True] * 5))
        pd.testing.assert_series_equal(is_valid['table_1'], pd.Series([True] * 3))
        pd.testing.assert_series_equal(is_valid['table_2'], pd.Series([True] * 2))

    def test__is_valid_with_mismatched_key_types(self, data, metadata, constraint):
        """Test ``_is_valid`` flags a key that does not have the type of the main key."""
        # Run
        is_valid = constraint._is_valid(data, metadata)

        # Assert
        pd.testing.assert_series_equal(is_valid['table_1'], pd.Series([True] * 3))
        pd.testing.assert_series_equal(is_valid['table_2'], pd.Series([False] * 2))

    def test_get_score(self, data, metadata, constraint):
        """Test ``get_score`` returns the proportion of valid rows."""
        # Setup
        data['table_2']['col_4'] = [2, 4]

        # Run & Assert
        assert constraint.get_score(data, metadata) == 1.0

    def test_get_score_invalid_data(self, data, metadata, constraint):
        """Test ``get_score`` counts the rows of every table involved."""
        # Setup
        data['table_2']['col_4'] = [2, 4]
        data['table_1']['col_1'] = [1, 2, 5]

        # Run & Assert
        assert constraint.get_score(data, metadata) == 0.9

    def test_get_score_empty_tables(self, data, metadata, constraint):
        """Test ``get_score`` returns NaN when there are no rows to check."""
        # Setup
        data = {table: table_data.iloc[:0] for table, table_data in data.items()}

        # Run & Assert
        assert pd.isna(constraint.get_score(data, metadata))
