"""Denormalized Table Constraint."""

from sdmetrics.multi_table.statistical.constraints.base import BaseConstraint
from sdmetrics.multi_table.statistical.constraints.utils import _get_table_to_valid_rows

class DenormalizedTable(BaseConstraint):

    _is_single_table = False

    def _is_valid(self, data, metadata=None):
        """Check that the data is valid.

        A row is considered invalid if the value in any column in denormalized_column_names
        does not match the value for other instances of the same key.
        """
        table = data[self.table_name]
        table_to_valid_rows = _get_table_to_valid_rows(data)
        if len(table) == 0:
            return table_to_valid_rows

        counts_per_row = table.groupby(self.denorm_pk, dropna=False)[self.denorm_columns].transform(
            lambda col: col.nunique(dropna=False)
        )

        row_invalid = (counts_per_row > 1).any(axis=1)
        if row_invalid.any():
            table_to_valid_rows[self.table_name].loc[row_invalid] = False

        return table_to_valid_rows
