import re
from copy import deepcopy

import numpy as np
import pandas as pd
import pytest

from sdmetrics.multi_table.statistical.constraints import SelfReferentialHierarchy
from sdmetrics.multi_table.statistical.constraints.error import ConstraintNotApplicableError


@pytest.fixture
def data_and_metadata():
    metadata = {
        'tables': {
            'main_table': {
                'columns': {
                    'base_id': {'sdtype': 'id', 'regex_format': r'\d{1}'},
                    'base_pii': {'sdtype': 'ssn'},
                    'parent_id': {'sdtype': 'id', 'regex_format': r'\d{1}'},
                    'parent_id_self': {'sdtype': 'id', 'regex_format': r'\d{1}'},
                    'parent_id_missing': {'sdtype': 'id', 'regex_format': r'\d{1}'},
                    'parent_pii': {'sdtype': 'ssn'},
                    'parent_pii_missing': {'sdtype': 'ssn'},
                    'root_id': {'sdtype': 'id'},
                    'grandparent_id': {'sdtype': 'id'},
                    'latitude1': {'sdtype': 'latitude'},
                    'longitude1': {'sdtype': 'longitude'},
                    'latitude2': {'sdtype': 'latitude'},
                    'longitude2': {'sdtype': 'longitude'},
                    'numeric': {'sdtype': 'numerical'},
                },
            },
            'other_table': {
                'columns': {
                    'base_column': {'sdtype': 'id'},
                    'column_3': {'sdtype': 'numerical'},
                    'column_4': {'sdtype': 'categorical'},
                },
                'primary_key': 'base_column',
            },
        },
        'relationships': [],
    }

    data = {
        'main_table': pd.DataFrame({
            'base_id': [1, 2, 3, 4, 5, 6],
            'base_pii': ['id1', 'id2', 'id3', 'id4', 'id5', 'id6'],
            'parent_id': [np.nan, 1, 2, 3, 4, 5],
            'parent_id_self': [1, 1, 2, 3, 4, 5],
            'parent_id_missing': [0, 1, 2, 3, 4, 5],
            'parent_pii': [None, 'id1', 'id2', 'id3', 'id4', 'id5'],
            'parent_pii_missing': ['id0', 'id1', 'id2', 'id3', 'id4', 'id5'],
            'latitude1': [37.7749, 37.7749, 37.7749, 37.7749, 37.7749, 37.7749],
            'longitude1': [-122.4194, -122.4194, -122.4194, -122.4194, -122.4194, -122.4194],
            'latitude2': [37.7749, 37.7749, 37.7749, 37.7749, 37.7749, 37.7749],
            'longitude2': [-122.4194, -122.4194, -122.4194, -122.4194, -122.4194, -122.4194],
            'numeric': [1.0, 1.0, 3.4, 3.14159, -98.2, 42.0],
        }),
        'other_table': pd.DataFrame({
            'base_column': [5, 6, 7, 8],
            'column_3': [1000, 2000, 3000, 4000],
            'column_4': ['X', 'Y', 'Z', 'W'],
        }),
    }

    return deepcopy(data), deepcopy(metadata)


@pytest.fixture()
def data_and_metadata_with_loop():
    table_name = 'employees'
    data = {
        table_name: pd.DataFrame({
            'Employee ID': ['A', 'B', 'C'],
            'Manager ID': ['B', 'C', 'A'],
        })
    }
    metadata = {
        'tables': {
            table_name: {
                'primary_key': 'Employee ID',
                'columns': {
                    'Employee ID': {'sdtype': 'id', 'regex_format': '[A-Z]{1,2}'},
                    'Manager ID': {'sdtype': 'id', 'regex_format': '[A-Z]{1,2}'},
                },
            }
        }
    }
    return deepcopy(data), deepcopy(metadata)


@pytest.fixture
def constraint():
    return SelfReferentialHierarchy(
        table_name='main_table',
        base_column_name='base_id',
        parent_column_name='parent_id',
    )


class TestSelfReferentialHierarchy:
    def test___init__(self):
        """Test the ``__init__`` method sets the parameters."""
        # Run
        instance = SelfReferentialHierarchy(
            table_name='main_table',
            base_column_name='base_id',
            parent_column_name='parent_id',
        )

        # Assert
        assert instance.table_name == 'main_table'
        assert instance._base_column == 'base_id'
        assert instance._parent_column == 'parent_id'
        assert instance._grandparent_column is None
        assert instance._root_column is None
        assert instance._scaling_method == 'branch'

    def test___init__with_optional_parameters(self):
        """Test the ``__init__`` method sets the optional parameters."""
        # Run
        instance = SelfReferentialHierarchy(
            table_name='main_table',
            base_column_name='base_id',
            parent_column_name='parent_id',
            grandparent_column_name='grandparent_id',
            root_column_name='root_id',
            scaling_method='depth',
        )

        # Assert
        assert instance._grandparent_column == 'grandparent_id'
        assert instance._root_column == 'root_id'
        assert instance._scaling_method == 'depth'

    def test___init__invalid_parameters(self):
        """Test the ``__init__`` method errors with invalid arguments."""
        # Setup
        parameters = {
            'table_name': 'main_table',
            'base_column_name': 'base_id',
            'parent_column_name': 'parent_id',
        }
        err_msg = re.escape(
            'The `table_name`, `base_column_name` and `parent_column_name` must be all be strings.'
        )

        # Run and Assert
        with pytest.raises(TypeError, match=err_msg):
            SelfReferentialHierarchy(**{**parameters, 'table_name': 1})

        with pytest.raises(TypeError, match=err_msg):
            SelfReferentialHierarchy(**{**parameters, 'base_column_name': ['base_id']})

        with pytest.raises(TypeError, match=err_msg):
            SelfReferentialHierarchy(**{**parameters, 'parent_column_name': 1})

    def test___init__invalid_optional_parameters(self):
        """Test the ``__init__`` method errors with invalid optional arguments."""
        # Setup
        parameters = {
            'table_name': 'main_table',
            'base_column_name': 'base_id',
            'parent_column_name': 'parent_id',
        }
        err_msg = re.escape(
            'The `grandparent_column_name` and `root_column_name` must be all be strings or `None`.'
        )

        # Run and Assert
        with pytest.raises(TypeError, match=err_msg):
            SelfReferentialHierarchy(**{**parameters, 'grandparent_column_name': 1})

        with pytest.raises(TypeError, match=err_msg):
            SelfReferentialHierarchy(**{**parameters, 'root_column_name': 1})

    def test___init__same_base_and_parent_column(self):
        """Test the ``__init__`` method errors if the base and the parent are the same."""
        # Setup
        err_msg = re.escape(
            'The `base_column_name` and `parent_column_name` must be different columns.'
        )

        # Run and Assert
        with pytest.raises(ValueError, match=err_msg):
            SelfReferentialHierarchy(
                table_name='main_table',
                base_column_name='base_id',
                parent_column_name='base_id',
            )

    def test___init__invalid_scaling_method(self):
        """Test the ``__init__`` method errors with an unknown scaling method."""
        # Setup
        err_msg = re.escape(
            "Unrecognized scaling_method 'wide'. The scaling method must be one of "
            "'branch', 'depth' or 'multiply'."
        )

        # Run and Assert
        with pytest.raises(ValueError, match=err_msg):
            SelfReferentialHierarchy(
                table_name='main_table',
                base_column_name='base_id',
                parent_column_name='parent_id',
                scaling_method='wide',
            )

    def test__validate_data_missing_table(self, data_and_metadata, constraint):
        """Test ``_validate_data`` errors if the table is not in the data."""
        # Setup
        data, metadata = data_and_metadata
        del data['main_table']
        expected_error = re.escape("The table 'main_table' is missing from the data.")

        # Run & Assert
        with pytest.raises(ConstraintNotApplicableError, match=expected_error):
            constraint._validate_data(data, metadata)

    def test__validate_data_missing_columns(self, data_and_metadata, constraint):
        """Test ``_validate_data`` errors if a referenced column is not in the table."""
        # Setup
        data, metadata = data_and_metadata
        del data['main_table']['parent_id']
        expected_error = re.escape(
            "The column(s) 'parent_id' are missing from the table 'main_table'."
        )

        # Run & Assert
        with pytest.raises(ConstraintNotApplicableError, match=expected_error):
            constraint._validate_data(data, metadata)

    def test__validate_data_missing_optional_columns(self, data_and_metadata):
        """Test ``_validate_data`` also checks the optional columns."""
        # Setup
        data, metadata = data_and_metadata
        instance = SelfReferentialHierarchy(
            table_name='main_table',
            base_column_name='base_id',
            parent_column_name='parent_id',
            grandparent_column_name='grandparent_id',
            root_column_name='root_id',
        )
        expected_error = re.escape(
            "The column(s) 'grandparent_id', 'root_id' are missing from the table 'main_table'."
        )

        # Run & Assert
        with pytest.raises(ConstraintNotApplicableError, match=expected_error):
            instance._validate_data(data, metadata)

    def test__is_valid(self, data_and_metadata, constraint):
        """Test ``_is_valid`` considers every row of a well formed hierarchy valid."""
        # Setup
        data, metadata = data_and_metadata

        # Run
        is_valid = constraint._is_valid(data, metadata)

        # Assert
        pd.testing.assert_series_equal(is_valid['main_table'], pd.Series([True] * 6))
        pd.testing.assert_series_equal(is_valid['other_table'], pd.Series([True] * 4))

    def test__is_valid_self_referencing_root(self, data_and_metadata):
        """Test ``_is_valid`` accepts a root that references itself."""
        # Setup
        data, metadata = data_and_metadata
        instance = SelfReferentialHierarchy(
            table_name='main_table',
            base_column_name='base_id',
            parent_column_name='parent_id_self',
        )

        # Run
        is_valid = instance._is_valid(data, metadata)

        # Assert
        pd.testing.assert_series_equal(is_valid['main_table'], pd.Series([True] * 6))

    def test__is_valid_parent_missing_from_the_table(self, data_and_metadata):
        """Test ``_is_valid`` accepts a parent that no row of the table defines."""
        # Setup
        data, metadata = data_and_metadata
        instance = SelfReferentialHierarchy(
            table_name='main_table',
            base_column_name='base_id',
            parent_column_name='parent_id_missing',
        )

        # Run
        is_valid = instance._is_valid(data, metadata)

        # Assert
        pd.testing.assert_series_equal(is_valid['main_table'], pd.Series([True] * 6))

    def test__is_valid_with_pii_columns(self, data_and_metadata):
        """Test ``_is_valid`` also follows a hierarchy that is built on a PII column."""
        # Setup
        data, metadata = data_and_metadata
        instance = SelfReferentialHierarchy(
            table_name='main_table',
            base_column_name='base_pii',
            parent_column_name='parent_pii',
        )

        # Run
        is_valid = instance._is_valid(data, metadata)

        # Assert
        pd.testing.assert_series_equal(is_valid['main_table'], pd.Series([True] * 6))

    def test__is_valid_with_repeated_base_value(self, data_and_metadata, constraint):
        """Test ``_is_valid`` flags a row that repeats the base value of an earlier row."""
        # Setup
        data, metadata = data_and_metadata
        data['main_table'].loc[5, 'base_id'] = 5

        # Run
        is_valid = constraint._is_valid(data, metadata)

        # Assert
        expected = pd.Series([True, True, True, True, True, False])
        pd.testing.assert_series_equal(is_valid['main_table'], expected)

    def test__is_valid_with_cycle(self, data_and_metadata_with_loop):
        """Test ``_is_valid`` flags every row that goes through a cycle."""
        # Setup
        data, metadata = data_and_metadata_with_loop
        instance = SelfReferentialHierarchy(
            table_name='employees',
            base_column_name='Employee ID',
            parent_column_name='Manager ID',
        )

        # Run
        is_valid = instance._is_valid(data, metadata)

        # Assert
        pd.testing.assert_series_equal(is_valid['employees'], pd.Series([False] * 3))

    def test__is_valid_null_base(self, data_and_metadata, constraint):
        """Test ``_is_valid`` flags a row that has no value in the base column."""
        # Setup
        data, metadata = data_and_metadata
        data['main_table']['base_id'] = [1.0, 2.0, np.nan, 4.0, 5.0, 6.0]

        # Run
        is_valid = constraint._is_valid(data, metadata)

        # Assert
        expected = pd.Series([True, True, False, True, True, True])
        pd.testing.assert_series_equal(is_valid['main_table'], expected)

    def test__is_valid_with_grandparent_column(self, data_and_metadata):
        """Test ``_is_valid`` checks the grandparent column against the hierarchy."""
        # Setup
        data, metadata = data_and_metadata
        data['main_table']['grandparent_id'] = [np.nan, np.nan, 1.0, 2.0, 3.0, 4.0]
        instance = SelfReferentialHierarchy(
            table_name='main_table',
            base_column_name='base_id',
            parent_column_name='parent_id',
            grandparent_column_name='grandparent_id',
        )

        # Run
        is_valid = instance._is_valid(data, metadata)

        # Assert
        pd.testing.assert_series_equal(is_valid['main_table'], pd.Series([True] * 6))

    def test__is_valid_with_invalid_grandparent(self, data_and_metadata):
        """Test ``_is_valid`` flags a grandparent that is not the parent of the parent."""
        # Setup
        data, metadata = data_and_metadata
        data['main_table']['grandparent_id'] = [np.nan, 6.0, 1.0, 2.0, 6.0, 4.0]
        instance = SelfReferentialHierarchy(
            table_name='main_table',
            base_column_name='base_id',
            parent_column_name='parent_id',
            grandparent_column_name='grandparent_id',
        )

        # Run
        is_valid = instance._is_valid(data, metadata)

        # Assert
        expected = pd.Series([True, False, True, True, False, True])
        pd.testing.assert_series_equal(is_valid['main_table'], expected)

    def test__is_valid_with_root_column(self, data_and_metadata):
        """Test ``_is_valid`` checks the root column against the hierarchy."""
        # Setup
        data, metadata = data_and_metadata
        data['main_table']['root_id'] = [1, 1, 1, 1, 1, 1]
        instance = SelfReferentialHierarchy(
            table_name='main_table',
            base_column_name='base_id',
            parent_column_name='parent_id',
            root_column_name='root_id',
        )

        # Run
        is_valid = instance._is_valid(data, metadata)

        # Assert
        pd.testing.assert_series_equal(is_valid['main_table'], pd.Series([True] * 6))

    def test__is_valid_with_invalid_root(self, data_and_metadata):
        """Test ``_is_valid`` flags a root that is not the top of the hierarchy."""
        # Setup
        data, metadata = data_and_metadata
        data['main_table']['root_id'] = [1, 1, 9, 1, 1, 9]
        instance = SelfReferentialHierarchy(
            table_name='main_table',
            base_column_name='base_id',
            parent_column_name='parent_id',
            root_column_name='root_id',
        )

        # Run
        is_valid = instance._is_valid(data, metadata)

        # Assert
        expected = pd.Series([True, True, False, True, True, False])
        pd.testing.assert_series_equal(is_valid['main_table'], expected)

    def test__is_valid_other_data(self):
        """Test that the right rows are marked as invalid."""
        # Setup
        data = {
            'main_table': pd.DataFrame({
                'base': [1, 2, 3, 4, 5, 6, 7, 8, 9],
                'employee': [1, 2, 3, 4, 5, 6, 5, 8, np.nan],
                'manager': [np.nan, 4, 2, 3, 1, 5, 1, 10, 6],
                'numeric': [1.0, 1.0, 3.4, 3.14159, -98.2, 42.0, 36.4, 18.0, 13.6],
            }),
            'other_table': pd.DataFrame({
                'base_column': [5, 6, 7, 8],
                'column_3': [1000, 2000, 3000, 4000],
                'column_4': ['X', 'Y', 'Z', 'W'],
            }),
        }
        table_name = 'main_table'
        base_column_name = 'employee'
        parent_column_name = 'manager'
        instance = SelfReferentialHierarchy(
            table_name=table_name,
            base_column_name=base_column_name,
            parent_column_name=parent_column_name,
        )

        # Run
        valid_rows = instance._is_valid(data)

        # Assert
        expected = {
            'main_table': pd.Series([True, False, False, False, True, True, False, True, False]),
            'other_table': pd.Series([True] * 4),
        }
        for table, data in expected.items():
            pd.testing.assert_series_equal(data, valid_rows[table])

    def test_get_score(self, data_and_metadata, constraint):
        """Test ``get_score`` returns the proportion of valid rows."""
        # Setup
        data, metadata = data_and_metadata

        # Run & Assert
        assert constraint.get_score(data, metadata) == 1.0

    def test_get_score_invalid_data(self, data_and_metadata_with_loop):
        """Test ``get_score`` returns the proportion of rows that respect the hierarchy."""
        # Setup
        data, metadata = data_and_metadata_with_loop
        instance = SelfReferentialHierarchy(
            table_name='employees',
            base_column_name='Employee ID',
            parent_column_name='Manager ID',
        )

        # Run & Assert
        assert instance.get_score(data, metadata) == 0.0

    def test_get_score_empty_table(self, data_and_metadata, constraint):
        """Test ``get_score`` returns NaN when there are no rows to check."""
        # Setup
        data, metadata = data_and_metadata
        data = {table: table_data.iloc[:0] for table, table_data in data.items()}

        # Run & Assert
        assert pd.isna(constraint.get_score(data, metadata))
