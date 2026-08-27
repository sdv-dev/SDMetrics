"""Polymorphic Relationship Constraint."""

import itertools
from collections import defaultdict

import pandas as pd

from sdmetrics.multi_table.statistical.constraints._utils import (
    _cast_to_iterable,
    _create_unique_name,
    _get_primary_key,
    _get_table_to_valid_rows,
    _is_list_of_type,
)
from sdmetrics.multi_table.statistical.constraints.base import BaseConstraint
from sdmetrics.multi_table.statistical.constraints.error import ConstraintNotApplicableError


class PolymorphicRelationship(BaseConstraint):
    """Polymorphic relationship constraint.

    This constraint handles the case where a single column acts as a foreign key into
    multiple possible tables, depending on either (1) the format of the foreign key
    itself or (2) the value in another column.

    Args:
        table_name (str)
            The name of the table that contains the shared foreign key.
        foreign_key (str or list[str])
            The name of the shared foreign key column that is present in the table.
        parent_table_names (list[str])
            A list of table names that the foreign key values refer to.
        type_column_name (str, optional)
            The name of the categorical column in the table that sets which table the
            foreign key references. If None, attempts to detect the parent table from the
            foreign key value. Defaults to None.
        type_value_to_table (dict, optional)
            A map of category values in the type column to the parent table being referenced.
            If None and `type_column_name` is passed, the table name is used as the type value
            for each parent. Defaults to None.
    """

    _is_single_table = False

    def _validate_polymorphic_relationship_inputs(
        self,
        table_name,
        foreign_key,
        parent_table_names,
        type_column_name,
        type_value_to_table,
    ):
        if not isinstance(table_name, str):
            raise TypeError('`table_name` must be a string.')

        if not isinstance(foreign_key, str) and not _is_list_of_type(foreign_key):
            raise TypeError('`foreign_key` must be a string or a list of strings.')

        if not _is_list_of_type(parent_table_names, str):
            raise TypeError('`parent_table_names` must be a list of strings.')
        elif table_name in parent_table_names:
            raise ConstraintNotApplicableError(
                f"Table name '{table_name}' cannot also be in `parent_table_names`."
            )

        if type_column_name is not None and not isinstance(type_column_name, str):
            raise TypeError('`type_column_name` must be a string or None.')

        if type_column_name in _cast_to_iterable(foreign_key):
            raise ValueError('`foreign_key` and `type_column_name` must be different columns.')

        if type_value_to_table is not None:
            if not isinstance(type_value_to_table, dict):
                raise TypeError('`type_value_to_table` must be a dict or `None`.')

            extra_tables = set(type_value_to_table.values()) - set(parent_table_names)
            missing_tables = set(parent_table_names) - set(type_value_to_table.values())
            if extra_tables:
                extra = "', '".join(list(extra_tables))
                raise ValueError(
                    f"Table(s) '{extra}' in `type_values_to_table` not found "
                    'in `parent_table_names` list.'
                )
            if missing_tables:
                missing = "', '".join(list(missing_tables))
                raise ValueError(
                    f"Table(s) '{missing}' in `parent_table_names` do not have any "
                    'type value associated with them in `type_values_to_table`.'
                )

    def _get_parent_type_dicts(self):
        type_value_to_parent = self.type_value_to_parent
        parent_to_type_values = defaultdict(list)
        if self.type_value_to_parent is not None:
            for value, parent in self.type_value_to_parent.items():
                parent_to_type_values[parent].append(value)
        else:
            type_value_to_parent = {parent: parent for parent in self.parent_tables}
            parent_to_type_values = {parent: [parent] for parent in self.parent_tables}

        return parent_to_type_values, type_value_to_parent

    def __init__(
        self,
        table_name,
        foreign_key,
        parent_table_names,
        type_column_name=None,
        type_value_to_table=None,
    ):
        super().__init__()

        self._validate_polymorphic_relationship_inputs(
            table_name,
            foreign_key,
            parent_table_names,
            type_column_name,
            type_value_to_table,
        )
        self.table_name = table_name
        self.foreign_key = foreign_key
        self._num_fk_cols = 1 if isinstance(foreign_key, str) else len(self.foreign_key)
        self.parent_tables = parent_table_names
        self.type_column = type_column_name
        self.type_value_to_table = type_value_to_table

        self.type_value_to_parent = type_value_to_table if type_column_name else None
        self._parent_to_type_values, self._type_value_to_parent = self._get_parent_type_dicts()

    def _validate_data(self, data, metadata=None):
        """Check that every table and all the referenced columns exist in the data."""
        if self.table_name not in data:
            raise ConstraintNotApplicableError(
                f"The table '{self.table_name}' is missing from the data."
            )

        columns = data[self.table_name].columns
        table_columns = [self.foreign_key]
        if self.type_column is not None:
            table_columns.append(self.type_column)

        missing_columns = [
            column_name for column_name in table_columns if column_name not in columns
        ]
        if missing_columns:
            missing_columns = "', '".join(missing_columns)
            raise ConstraintNotApplicableError(
                f"The column(s) '{missing_columns}' are missing from the table '{self.table_name}'."
            )

        for parent_table_name in self.parent_tables:
            if parent_table_name not in data:
                raise ConstraintNotApplicableError(
                    f"The table '{parent_table_name}' is missing from the data."
                )

            primary_key = _get_primary_key(metadata, parent_table_name)
            if primary_key not in data[parent_table_name].columns:
                raise ConstraintNotApplicableError(
                    f"The column(s) '{primary_key}' are missing from the table "
                    f"'{parent_table_name}'."
                )

    def _get_foreign_key_groups(self, data, metadata):
        table_data = data[self.table_name]
        type_column = self.type_column
        parent_to_types = self._parent_to_type_values
        child_groups = {}
        foreign_key = _cast_to_iterable(self.foreign_key)
        if self.type_column is not None:
            for parent, type_values in parent_to_types.items():
                child_rows_mask = table_data[type_column].isin(type_values)
                child_groups[parent] = table_data[foreign_key][child_rows_mask]

        else:
            child_dtypes = list(table_data[foreign_key].dtypes)
            for parent in self.parent_tables:
                parent_pk = _cast_to_iterable(_get_primary_key(metadata, parent))
                parent_pk_values = data[parent][parent_pk].astype({
                    pk_col: dtype for pk_col, dtype in zip(parent_pk, child_dtypes)
                })
                indicator_col = _create_unique_name('_merge', foreign_key + parent_pk)
                merged_child = table_data[foreign_key].merge(
                    parent_pk_values,
                    left_on=foreign_key,
                    right_on=parent_pk,
                    how='left',
                    indicator=indicator_col,
                )
                child_groups[parent] = table_data[merged_child[indicator_col] == 'both'][
                    foreign_key
                ]

        return child_groups

    def _validate_type_column(self, table_data):
        type_column = table_data[self.type_column]
        type_values = (
            self.parent_tables
            if not self.type_value_to_parent
            else self.type_value_to_parent.keys()
        )
        bad_type_values = ~(type_column.isin(type_values) | type_column.isna())

        return ~bad_type_values

    def _validate_parent_primary_keys(self, data, metadata):
        is_valid_pk = {
            parent: pd.Series(True, index=data[parent].index) for parent in self.parent_tables
        }
        primary_keys = {parent: _get_primary_key(metadata, parent) for parent in self.parent_tables}
        overlapping_parents = {}
        for parent1, parent2 in itertools.combinations(primary_keys, 2):
            table1_pk = _cast_to_iterable(primary_keys[parent1])
            table2_pk = _cast_to_iterable(primary_keys[parent2])
            indicator_col = _create_unique_name('_merge', table1_pk + table2_pk)
            invalid_parent2_mask = (
                data[parent2][table2_pk]
                .astype('object')
                .merge(
                    data[parent1][table1_pk].astype('object'),
                    left_on=table2_pk,
                    right_on=table1_pk,
                    how='left',
                    indicator=indicator_col,
                )
                .set_index(data[parent2].index)[indicator_col]
                == 'both'
            )
            if any(invalid_parent2_mask):
                overlapping_parents[(parent1, parent2)] = invalid_parent2_mask

        for (_, table2), invalid_pk_mask in overlapping_parents.items():
            is_valid_pk[table2] &= ~invalid_pk_mask

        return is_valid_pk

    def _validate_polymorphic_relationship_with_data(self, data, metadata):
        table_to_valid_rows = _get_table_to_valid_rows(data)
        table_data = data[self.table_name]
        valid_rows = table_to_valid_rows[self.table_name]
        referenced = pd.Series(False, index=table_data.index)
        referenced[table_data[_cast_to_iterable(self.foreign_key)].isna().all(axis=1)] = True

        if self.type_column:
            valid_rows[~self._validate_type_column(table_data)] = False
        else:
            table_to_valid_rows.update(self._validate_parent_primary_keys(data, metadata))
            valid_rows = table_to_valid_rows[self.table_name]

        child_groups = self._get_foreign_key_groups(data, metadata)
        for parent, child_ids in child_groups.items():
            referenced[child_ids.index] = True
            primary_key = _cast_to_iterable(_get_primary_key(metadata, parent))
            foreign_key = list(child_ids.columns)
            indicator_col = _create_unique_name('_merge', primary_key + foreign_key)
            unknown_keys_mask = (
                child_ids
                .dropna(how='all')
                .merge(
                    data[parent][primary_key],
                    left_on=foreign_key,
                    right_on=primary_key,
                    how='left',
                    indicator=indicator_col,
                )
                .set_index(child_ids.dropna(how='all').index)[indicator_col]
                == 'left_only'
            )
            if any(unknown_keys_mask):
                valid_rows[valid_rows & unknown_keys_mask] = False

        if not all(referenced):
            valid_rows[~referenced] = False

        table_to_valid_rows[self.table_name] = valid_rows
        return table_to_valid_rows

    def _is_valid(self, data, metadata):
        """Return whether or not each row in the data dictionary is valid for the constraint.

        Args:
            data (dict[str, pd.DataFrame]):
                Table data.
            metadata (sdv.Metadata):
                Metadata for the data.

        Returns:
            dict[str, pd.Series]:
                A dictionary mapping the table name to a Series where each row is=True or False
                depending on if it's valid.
        """
        table_to_valid_rows = self._validate_polymorphic_relationship_with_data(data, metadata)

        return table_to_valid_rows
