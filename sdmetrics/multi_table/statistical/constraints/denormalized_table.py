"""Denormalized Table Constraint."""

from sdmetrics.multi_table.statistical.constraints.base import BaseConstraint
from sdmetrics.multi_table.statistical.constraints.error import ConstraintNotApplicableError
from sdmetrics.multi_table.statistical.constraints._utils import _get_table_to_valid_rows


class DenormalizedTable(BaseConstraint):
    """Constraint for a table that contains denormalized columns.

    A denormalized table repeats the columns of a parent table on every row that
    references the same parent. The data adheres to this constraint when every row
    that shares a ``denormalized_primary_key`` value also shares the exact same
    values for all the ``denormalized_column_names``.

    Args:
        table_name (str):
            The name of the denormalized table.
        denormalized_primary_key (str):
            The name of the column that contains the primary key of the parent table.
        denormalized_column_names (list[str] or None):
            The names of the columns that come from the parent table. If ``None`` or
            empty, there is nothing to check and every row is considered valid.
    """

    def __init__(self, table_name, denormalized_primary_key, denormalized_column_names=None):
        super().__init__()
        if not isinstance(table_name, str):
            raise ValueError("The 'table_name' parameter must be a string.")

        if not isinstance(denormalized_primary_key, str):
            raise ValueError("The 'denormalized_primary_key' parameter must be a string.")

        if denormalized_column_names is None:
            denormalized_column_names = []

        is_list_of_strings = isinstance(denormalized_column_names, list) and all(
            isinstance(column_name, str) for column_name in denormalized_column_names
        )
        if not is_list_of_strings:
            raise ValueError("The 'denormalized_column_names' parameter must be a list of strings.")

        if denormalized_primary_key in denormalized_column_names:
            raise ValueError(
                f"The column '{denormalized_primary_key}' cannot be both the "
                "'denormalized_primary_key' and one of the 'denormalized_column_names'."
            )

        self.table_name = table_name
        self.denormalized_primary_key = denormalized_primary_key
        self.denormalized_column_names = list(denormalized_column_names)

    def _validate_data(self, data, metadata=None):
        """Check that the table and all the referenced columns exist in the data."""
        if self.table_name not in data:
            raise ConstraintNotApplicableError(
                f"The table '{self.table_name}' is missing from the data."
            )

        columns = data[self.table_name].columns
        missing_columns = [
            column_name
            for column_name in [self.denormalized_primary_key, *self.denormalized_column_names]
            if column_name not in columns
        ]
        if missing_columns:
            missing_columns = "', '".join(missing_columns)
            raise ConstraintNotApplicableError(
                f"The column(s) '{missing_columns}' are missing from the table '{self.table_name}'."
            )

    def _is_valid(self, data, metadata=None):
        """Check that the data is valid.

        A row is considered invalid if the value in any column in denormalized_column_names
        does not match the value for other instances of the same key.

        Args:
            data (dict[str, pandas.DataFrame]):
                The data dictionary.
            metadata (dict):
                Metadata as a dictionary.

        Returns:
            dict[str, pandas.Series]:
                Whether each row is valid.
        """
        table = data[self.table_name]
        table_to_valid_rows = _get_table_to_valid_rows(data)
        if not self.denormalized_column_names or table.empty:
            return table_to_valid_rows

        counts_per_row = table.groupby(self.denormalized_primary_key, dropna=False)[
            self.denormalized_column_names
        ].transform(lambda col: col.nunique(dropna=False))

        row_invalid = (counts_per_row > 1).any(axis=1)
        if row_invalid.any():
            table_to_valid_rows[self.table_name].loc[row_invalid] = False

        return table_to_valid_rows
