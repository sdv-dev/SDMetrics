"""Carry Over Columns Constraint."""

from copy import deepcopy

from sdmetrics.multi_table.statistical.constraints.base import BaseConstraint
from sdmetrics.multi_table.statistical.constraints.error import ConstraintNotApplicableError
from sdmetrics.multi_table.statistical.constraints.utils import (
    _get_table_to_valid_rows,
    _replace_nans_with_none,
)


def _validate_carry_over_columns(data, common_column_info):
    """Validate the CarryOverColumn constraint for the data.

    Validate that all carry over column rows share the same value for values across all
    'key_column_name' columns. Within a table, the carry over column's values should be
    consistent for the same value of the key column. Between tables, the carry over column's
    values should be consistent for the same value of the key column

    Args:
        data (dict[pd.DataFrame]):
            The data dictionary to validate.
        common_column_info (list[dict]):
            A list of dictionaries where each dictionary has the keys 'table_name',
            'carryover_column_name', and 'key_column_name'.
    """
    keys_to_column_info = {}
    key_value_pair = {}
    table_to_valid_rows = _get_table_to_valid_rows(data)
    for column_info in common_column_info:
        carry_over_column = column_info['carryover_column_name']
        key_column = column_info['key_column_name']
        table_name = column_info['table_name']
        table = deepcopy(data[table_name])
        carry_over_grouped_by_key = table.groupby(key_column, dropna=False)[[carry_over_column]]
        carry_over_unique_counts = carry_over_grouped_by_key.nunique(dropna=False)
        keys_with_over_1_value = (carry_over_unique_counts > 1).any(axis=1)
        inconsistent_vals = carry_over_unique_counts[keys_with_over_1_value].index
        if len(inconsistent_vals) > 0:
            table_to_valid_rows[table_name].loc[inconsistent_vals] = False

        table[key_column] = _replace_nans_with_none(table[key_column])
        table[carry_over_column] = _replace_nans_with_none(table[carry_over_column])
        mapping = table.set_index(key_column)[carry_over_column].to_dict()
        for key, value in mapping.items():
            existing_value = key_value_pair.get(key)
            if existing_value is None:
                key_value_pair[key] = value
                keys_to_column_info[key] = {
                    'table_name': table_name,
                    'carryover_column_name': carry_over_column,
                    'key_column_name': key_column,
                }
            elif existing_value != value:
                key_matches = table[key_column] == key
                carry_over_matches = table[carry_over_column] == value
                invalid_rows = table[key_matches & carry_over_matches]
                table_to_valid_rows[table_name].loc[invalid_rows.index] = False

    return table_to_valid_rows


class CarryOverColumns(BaseConstraint):
    """Constraint for a table that carries columns.

    The Carry Over Columns constraint that checks columns that were carried over
    from a parent table to a child table.

    Args:
        common_column_info (list[dict]):
            A list of dictionaries containing the following keys:
            - `table_name`: The name of the table.
            - `carryover_column_name`: The name of the column to carry over.
            - `key_column_name`: The name of the column to use as the shared key
               for the carried over column. Must be a PII or ID sdtype.
    """

    _is_single_table = False

    def __init__(self, common_column_info):
        super().__init__()

        expected_keys = {'table_name', 'carryover_column_name', 'key_column_name'}
        if not isinstance(common_column_info, list):
            raise TypeError('`common_column_info` must be a list.')

        for column_info in common_column_info:
            if not isinstance(column_info, dict):
                raise TypeError('Each element of `common_column_info` must be a dictionary.')
            if not set(column_info.keys()) == expected_keys:
                raise ValueError(
                    "Each element of `common_column_info` must have the keys 'table_name', "
                    "'carryover_column_name', and 'key_column_name'."
                )

            all_values_str = all(isinstance(column_info[key], str) for key in expected_keys)
            if not all_values_str:
                raise TypeError(
                    "The values of 'table_name', 'carryover_column_name', and 'key_column_name' "
                    'in each element of `common_column_info` must be strings.'
                )

        self.common_column_info = common_column_info
        self.table_name = None
        table_names = set(column_info['table_name'] for column_info in common_column_info)
        if len(table_names) == 1:
            # required to work in single table synthesizer
            self.table_name = table_names.pop()
            self._is_single_table = True

    def _validate_data(self, data, metadata=None):
        """Check that every table and all the referenced columns exist in the data."""
        for column_info in self.common_column_info:
            table_name = column_info['table_name']
            if table_name not in data:
                raise ConstraintNotApplicableError(
                    f"The table '{table_name}' is missing from the data."
                )

            columns = data[table_name].columns
            table_columns = [column_info['key_column_name'], column_info['carryover_column_name']]
            missing_columns = [
                column_name for column_name in table_columns if column_name not in columns
            ]
            if missing_columns:
                missing_columns = "', '".join(missing_columns)
                raise ConstraintNotApplicableError(
                    f"The column(s) '{missing_columns}' are missing from the table '{table_name}'."
                )

    def _is_valid(self, data, metadata=None):
        """Check that the data is valid.

        A row is considered invalid if the value of its carryover column does not match
        the value that other rows with the same key have, in any of the tables.
        """
        return _validate_carry_over_columns(data, self.common_column_info)
