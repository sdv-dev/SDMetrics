import re
from unittest.mock import Mock

import numpy as np
import pandas as pd
import pytest

from sdmetrics.multi_table.statistical.constraints import PolymorphicRelationship
from sdmetrics.multi_table.statistical.constraints.error import ConstraintNotApplicableError


@pytest.fixture
def data():
    return {
        'parent1': pd.DataFrame({'primary_key': [1, 2]}),
        'parent2': pd.DataFrame({'primary_key': [10, 20]}),
        'table': pd.DataFrame({
            'foreign_key': [1, 2, 10, 20, np.nan],
            'type': ['DEBIT', 'DEBIT', 'CREDIT', 'CREDIT', 'DEBIT'],
        }),
    }


@pytest.fixture
def metadata():
    return {
        'tables': {
            'table': {
                'columns': {'foreign_key': {'sdtype': 'id'}, 'type': {'sdtype': 'categorical'}}
            },
            'parent1': {'columns': {'primary_key': {'sdtype': 'id'}}, 'primary_key': 'primary_key'},
            'parent2': {'columns': {'primary_key': {'sdtype': 'id'}}, 'primary_key': 'primary_key'},
        }
    }


@pytest.fixture
def constraint():
    return PolymorphicRelationship(
        table_name='table',
        foreign_key='foreign_key',
        parent_table_names=['parent1', 'parent2'],
        type_column_name='type',
        type_value_to_table={'CREDIT': 'parent2', 'DEBIT': 'parent1'},
    )


class TestPolymorphicRelationship:
    def test___init__(self, constraint):
        """Test the ``__init__`` method sets the parameters."""
        # Assert
        assert constraint.table_name == 'table'
        assert constraint.foreign_key == 'foreign_key'
        assert constraint.parent_tables == ['parent1', 'parent2']
        assert constraint.type_column == 'type'
        assert constraint.type_value_to_table == {
            'CREDIT': 'parent2',
            'DEBIT': 'parent1',
        }

    def test___init__without_type_column(self):
        """Test the ``__init__`` method accepts a constraint without a type column."""
        # Run
        instance = PolymorphicRelationship(
            table_name='table',
            foreign_key='foreign_key',
            parent_table_names=['parent1'],
        )

        # Assert
        assert instance.type_column is None
        assert instance.type_value_to_table is None

    def test___init__invalid_parameters(self):
        """Test the ``__init__`` method errors with invalid arguments."""
        # Setup
        parameters = {
            'table_name': 'table',
            'foreign_key': 'foreign_key',
            'parent_table_names': ['parent1'],
        }

        # Run and Assert
        err_msg = re.escape('`table_name` must be a string.')
        with pytest.raises(TypeError, match=err_msg):
            PolymorphicRelationship(**{**parameters, 'table_name': 1})

        err_msg = re.escape('`foreign_key` must be a string or a list of strings.')
        with pytest.raises(TypeError, match=err_msg):
            PolymorphicRelationship(**{**parameters, 'foreign_key': 1})

        err_msg = re.escape('`parent_table_names` must be a list of strings.')
        with pytest.raises(TypeError, match=err_msg):
            PolymorphicRelationship(**{**parameters, 'parent_table_names': 'parent1'})

        err_msg = re.escape('`type_column_name` must be a string or None.')
        with pytest.raises(TypeError, match=err_msg):
            PolymorphicRelationship(**{**parameters, 'type_column_name': 1})

    def test___init__table_name_in_parent_table_names(self):
        """Test the ``__init__`` method errors if the table is one of its own parents."""
        # Setup
        err_msg = re.escape("Table name 'parent1' cannot also be in `parent_table_names`.")

        # Run and Assert
        with pytest.raises(ConstraintNotApplicableError, match=err_msg):
            PolymorphicRelationship(
                table_name='parent1',
                foreign_key='foreign_key',
                parent_table_names=['parent1'],
            )

    def test___init__type_column_is_the_foreign_key(self):
        """Test the ``__init__`` method errors if the type column is the foreign key."""
        # Setup
        err_msg = re.escape('`foreign_key` and `type_column_name` must be different columns.')

        # Run and Assert
        with pytest.raises(ValueError, match=err_msg):
            PolymorphicRelationship(
                table_name='table',
                foreign_key='foreign_key',
                parent_table_names=['parent1'],
                type_column_name='foreign_key',
            )

    def test___init__invalid_type_value_to_table(self):
        """Test the ``__init__`` method errors if the type mapping is inconsistent."""
        # Setup
        parameters = {
            'table_name': 'table',
            'foreign_key': 'foreign_key',
            'parent_table_names': ['parent1'],
        }

        # Run and Assert
        err_msg = re.escape('`type_value_to_table` must be a dict or `None`.')
        with pytest.raises(TypeError, match=err_msg):
            PolymorphicRelationship(**{
                **parameters,
                'type_column_name': 'type',
                'type_value_to_table': 'debit',
            })

        err_msg = re.escape(
            "Table(s) 'parent2' in `type_values_to_table` not found in `parent_table_names` list."
        )
        with pytest.raises(ValueError, match=err_msg):
            PolymorphicRelationship(**{
                **parameters,
                'type_column_name': 'type',
                'type_value_to_table': {'CREDIT': 'parent2'},
            })

        err_msg = re.escape(
            "Table(s) 'parent2' in `parent_table_names` do not have any type value "
            'associated with them in `type_values_to_table`.'
        )
        with pytest.raises(ValueError, match=err_msg):
            PolymorphicRelationship(**{
                **parameters,
                'parent_table_names': ['parent1', 'parent2'],
                'type_column_name': 'type',
                'type_value_to_table': {'DEBIT': 'parent1'},
            })

    def test__validate_data_missing_table(self, data, metadata, constraint):
        """Test ``_validate_data`` errors if a parent table is not in the data."""
        # Setup
        del data['parent1']
        expected_error = re.escape("The table 'parent1' is missing from the data.")

        # Run and Assert
        with pytest.raises(ConstraintNotApplicableError, match=expected_error):
            constraint._validate_data(data, metadata)

    def test__validate_data_missing_column(self, data, metadata, constraint):
        """Test ``_validate_data`` errors if the type column is not in the table."""
        # Setup
        del data['table']['type']
        expected_error = re.escape("The column(s) 'type' are missing from the table 'table'.")

        # Run and Assert
        with pytest.raises(ConstraintNotApplicableError, match=expected_error):
            constraint._validate_data(data, metadata)

    def test__validate_data_missing_primary_key_column(self, data, metadata, constraint):
        """Test ``_validate_data`` errors if the parent does not hold its primary key."""
        # Setup
        data['parent1']['id'] = data['parent1']['primary_key']
        del data['parent1']['primary_key']
        expected_error = re.escape(
            "The column(s) 'primary_key' are missing from the table 'parent1'."
        )

        # Run and Assert
        with pytest.raises(ConstraintNotApplicableError, match=expected_error):
            constraint._validate_data(data, metadata)

    def test__validate_data_missing_primary_key(self, data, metadata, constraint):
        """Test ``_validate_data`` errors if a parent has no primary key in the metadata."""
        # Setup
        del metadata['tables']['parent1']['primary_key']
        expected_error = re.escape(
            "The table 'parent1' does not have a primary key in the metadata."
        )

        # Run and Assert
        with pytest.raises(ConstraintNotApplicableError, match=expected_error):
            constraint._validate_data(data, metadata)

    def test__get_foreign_key_groups(self, metadata):
        """Test helper to group foreign keys by parent with a type column."""
        # Setup
        instance = PolymorphicRelationship(
            table_name='table',
            foreign_key='foreign_key',
            parent_table_names=['parent1', 'parent2'],
        )
        instance_type_col = PolymorphicRelationship(
            table_name='table',
            foreign_key='foreign_key',
            parent_table_names=['parent1', 'parent2'],
            type_column_name='type',
        )
        data = {
            'table': pd.DataFrame({
                'foreign_key': ['id0', 'id0', 'id0', 'id1', 'id2', 0, 0, 0, 2, 2],
                'type': ['parent1'] * 5 + ['parent2'] * 5,
            }),
            'parent1': pd.DataFrame({'primary_key': ['id0', 'id1', 'id2']}),
            'parent2': pd.DataFrame({'primary_key': [0, 1, 2]}),
        }

        # Run
        child_groups = instance._get_foreign_key_groups(data, metadata)
        child_groups_type_col = instance_type_col._get_foreign_key_groups(data, metadata)

        # Assert
        for result in (child_groups, child_groups_type_col):
            assert set(result.keys()) == {'parent1', 'parent2'}
            expected_parent1_group = pd.DataFrame({
                'foreign_key': ['id0', 'id0', 'id0', 'id1', 'id2']
            })
            pd.testing.assert_frame_equal(
                result['parent1'], expected_parent1_group, check_names=False
            )
            expected_parent2_group = pd.DataFrame(
                {'foreign_key': [0, 0, 0, 2, 2]}, index=[5, 6, 7, 8, 9], dtype='object'
            )

            pd.testing.assert_frame_equal(
                result['parent2'], expected_parent2_group, check_names=False
            )

    def test__validate_type_column(self):
        """Test type column validation."""
        instance = PolymorphicRelationship(
            table_name='table',
            foreign_key='fk',
            parent_table_names=['parent1', 'parent2'],
            type_column_name='type',
        )
        table_data = pd.DataFrame({
            'fk': ['id0', 'id0', 'id0', 'id1', 'id2', 0, 0, 0, 2, 2],
            'type': ['parent1'] * 5 + ['parent2'] * 4 + ['unknown'],
        })
        valid_type = ['parent1'] * 5 + ['parent2'] * 5

        # Run and Assert
        unknown_is_valid = instance._validate_type_column(table_data)

        instance.type_value_to_parent = {'parent1': 'parent1', 'parent2': 'parent2'}
        extra_is_valid = instance._validate_type_column(table_data)

        table_data['type'] = valid_type
        valid = instance._validate_type_column(table_data)

        # Assert
        expected_is_valid = pd.Series([True] * 9 + [False], name='type')
        pd.testing.assert_series_equal(extra_is_valid, expected_is_valid)
        pd.testing.assert_series_equal(unknown_is_valid, expected_is_valid)
        pd.testing.assert_series_equal(valid, pd.Series([True] * 10, name='type'))

    def test__validate_parent_primary_keys(self, metadata):
        """Test validating that primary key values do not overlap."""
        instance = PolymorphicRelationship(
            table_name='table',
            foreign_key='fk',
            parent_table_names=['parent1', 'parent2'],
        )
        data = {
            'table': pd.DataFrame({
                'fk': ['id0', 'id0', 'id0', 'id1', 'id2', 0, 0, 0, 2, 2],
                'type': ['parent1'] * 5 + ['parent2'] * 4 + ['unknown'],
            }),
            'parent1': pd.DataFrame({'primary_key': ['id0', 'id1', 'id2']}),
            'parent2': pd.DataFrame({'primary_key': ['id2', 'id3', 'id4']}),
        }

        # Run and Assert
        overlap_is_valid_dict = instance._validate_parent_primary_keys(data, metadata)

        # Assert
        assert set(overlap_is_valid_dict.keys()) == {'parent1', 'parent2'}
        assert all(overlap_is_valid_dict['parent1'])
        expected_is_valid_parent2 = pd.Series([False, True, True])
        pd.testing.assert_series_equal(
            overlap_is_valid_dict['parent2'], expected_is_valid_parent2, check_names=False
        )

    def test__validate_polymorphic_relationship_with_data(self, metadata):
        """Test validation with data without erroring."""
        instance = PolymorphicRelationship(
            table_name='table',
            foreign_key='foreign_key',
            parent_table_names=['parent1', 'parent2'],
            type_column_name='type',
        )
        instance._get_foreign_key_groups = Mock(
            return_value={
                'parent1': pd.DataFrame(
                    {'foreign_key': ['bad_key', 'id0', 'id1', 'id2']}, index=range(1, 5)
                ),
                'parent2': pd.DataFrame(
                    {'foreign_key': [0, 0, 0, 2, 'unknown']}, index=range(5, 10)
                ),
            }
        )
        data = {
            'table': pd.DataFrame({
                'foreign_key': ['id0', 'bad_key', 'id0', 'id1', 'id2', 0, 0, 0, 2, 'unknown'],
                'type': ['extra'] + ['parent1'] * 4 + ['parent2'] * 5,
            }),
            'parent1': pd.DataFrame({'primary_key': ['id0', 'id1', 'id2']}),
            'parent2': pd.DataFrame({'primary_key': [0, 1, 2]}),
        }

        # Run
        is_valid_dict = instance._validate_polymorphic_relationship_with_data(data, metadata)

        # Assert
        instance._get_foreign_key_groups.assert_called_once_with(data, metadata)
        expected_is_valid_table = pd.Series([False, False] + [True] * 7 + [False])
        for table, is_valid in is_valid_dict.items():
            if table == 'table':
                pd.testing.assert_series_equal(is_valid, expected_is_valid_table)
            else:
                assert all(is_valid)

    def test__is_valid(self, metadata):
        """Test ``_is_valid`` method."""
        # Setup
        instance = PolymorphicRelationship(
            table_name='table',
            foreign_key='foreign_key',
            parent_table_names=['parent1', 'parent2'],
        )
        data = {
            'table': pd.DataFrame({'foreign_key': list(range(5)) * 2}),
            'parent1': pd.DataFrame({
                'primary_key': [0, 1, 2],
            }),
            'parent2': pd.DataFrame({
                'primary_key': [3, 4],
            }),
        }

        # Run
        is_valid = instance._is_valid(data, metadata)

        # Assert
        assert set(is_valid.keys()) == {'table', 'parent1', 'parent2'}
        pd.testing.assert_series_equal(is_valid['table'], pd.Series([True] * 10))
        pd.testing.assert_series_equal(is_valid['parent1'], pd.Series([True] * 3))
        pd.testing.assert_series_equal(is_valid['parent2'], pd.Series([True] * 2))

    def test_get_score(self, data, metadata, constraint):
        """Test ``get_score`` returns the proportion of valid rows."""
        # Run and Assert
        assert constraint.get_score(data, metadata) == 1.0

    def test_get_score_invalid_data(self, data, metadata, constraint):
        """Test ``get_score`` counts the rows of every table involved."""
        # Setup
        data['table']['type'] = ['CREDIT', 'DEBIT', 'DEBIT', 'CREDIT', 'DEBIT']

        # Run and Assert
        assert constraint.get_score(data, metadata) == pytest.approx(7 / 9)

    def test_get_score_empty_tables(self, data, metadata, constraint):
        """Test ``get_score`` returns NaN when there are no rows to check."""
        # Setup
        data = {table: table_data.iloc[:0] for table, table_data in data.items()}

        # Run and Assert
        assert pd.isna(constraint.get_score(data, metadata))
