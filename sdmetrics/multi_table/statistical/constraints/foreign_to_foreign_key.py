"""Foreign To Foreign Key Constraint."""

import pandas as pd

from sdmetrics.multi_table.statistical.constraints._utils import (
    _get_row_tuples,
    _get_table_to_valid_rows,
    _validate_foreign_to_foreign_key_input,
)
from sdmetrics.multi_table.statistical.constraints.base import BaseConstraint
from sdmetrics.multi_table.statistical.constraints.error import ConstraintNotApplicableError


def _get_key_columns(foreign_key):
    """Return the list of columns that make up a foreign key."""
    if isinstance(foreign_key, tuple):
        return list(foreign_key)

    return [foreign_key]


class ForeignToForeignKey(BaseConstraint):
    """Constraint to check many-to-many foreign key relationships.

    Args:
        columns (list[dict]):
            A list of dictionaries, each specifying a foreign key from a table
            that is logically connected to others. Each dictionary should contain:
                - `'table_name' (str)`: The name of the table containing the foreign key.
                - `'foreign_key' (str | tuple[str])`: The name of the foreign key column, or a
                  tuple of column names if the foreign key is composite.
        foreign_key_generation (str):
            How to generate foreign key values. Must be on of `'new'` and `'reuse'`. If `'new'`,
            the synthetic data will create entirely new foreign key values that will be shared
            between the tables. If `'reuse'`, the same foreign key values will be reused from
            the original data. Defaults to `'new'`.
    """

    _is_single_table = False

    def __init__(self, columns, foreign_key_generation='new'):
        super().__init__()

        _validate_foreign_to_foreign_key_input(columns, foreign_key_generation)
        self.columns = columns
        self.foreign_key_generation = foreign_key_generation
        self._real_key_values = None

    def _get_scored_tables(self, metadata=None):
        return {column_info['table_name'] for column_info in self.columns}

    def _validate_data(self, data, metadata=None):
        """Check that every table and all the referenced columns exist in the data."""
        for column_info in self.columns:
            table_name = column_info['table_name']
            if table_name not in data:
                raise ConstraintNotApplicableError(
                    f"The table '{table_name}' is missing from the data."
                )

            columns = data[table_name].columns
            missing_columns = [
                column_name
                for column_name in _get_key_columns(column_info['foreign_key'])
                if column_name not in columns
            ]
            if missing_columns:
                missing_columns = "', '".join(missing_columns)
                raise ConstraintNotApplicableError(
                    f"The column(s) '{missing_columns}' are missing from the table '{table_name}'."
                )

    def _fit(self, data, metadata=None):
        """Learn the foreign key values of the real data when they must be reused.

        Args:
            data (dict[str, pandas.DataFrame]):
                A dictionary mapping each table name to its real data.
            metadata (dict):
                The multi table metadata.
        """
        if self.foreign_key_generation != 'reuse':
            return

        self._real_key_values = set()
        for column_info in self.columns:
            table_data = data[column_info['table_name']]
            key_columns = _get_key_columns(column_info['foreign_key'])
            self._real_key_values.update(_get_row_tuples(table_data, key_columns))

    def _is_valid(self, data, metadata=None):
        """Check that the data is valid.

        Args:
            data (dict[str, pandas.DataFrame]):
                The data dictionary.
            metadata (dict):
                Metadata as a dictionary.

        Returns:
            dict[str, pandas.Series]:
        """
        table_to_valid_rows = _get_table_to_valid_rows(data)
        if self.foreign_key_generation != 'reuse':
            return table_to_valid_rows

        if not self._fitted:
            raise ConstraintNotApplicableError(
                'ForeignToForeignKey constraint must be called with ``fit`` first.'
            )

        for column_info in self.columns:
            table_name = column_info['table_name']
            table_data = data[table_name]
            keys = _get_row_tuples(table_data, _get_key_columns(column_info['foreign_key']))
            is_valid = [
                all(value is None for value in key) or key in self._real_key_values for key in keys
            ]
            table_to_valid_rows[table_name] &= pd.Series(
                is_valid, index=table_data.index, dtype=bool
            )

        return table_to_valid_rows
