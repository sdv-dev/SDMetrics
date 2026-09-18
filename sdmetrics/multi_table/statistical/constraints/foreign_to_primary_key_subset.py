"""Foreign To Primary Key Subset Constraint."""

from sdmetrics.multi_table.statistical.constraints._utils import (
    _get_primary_key,
    _validate_foreign_to_primary_key_subset,
    _validate_foreign_to_primary_key_subset_input,
)
from sdmetrics.multi_table.statistical.constraints.base import BaseConstraint
from sdmetrics.multi_table.statistical.constraints.error import ConstraintNotApplicableError


class ForeignToPrimaryKeySubset(BaseConstraint):
    """Constraint for a foreign key that may only reference a subset of the parent rows.

    Args:
        parent_table_name (str):
            Name of the parent table.
        child_table_name (str):
            Name of the child table.
        child_foreign_key (str or list[str]):
            Name of the column (or list of column names for composite keys) in the child table
            that is a foreign key to the parent table.
        conditional_column_name (str):
            Name of the column in the parent table that defines the subset of valid primary key
            values.
        conditional_values (list):
            List of values in the ``conditional_column_name`` column that define the subset of
            valid primary key values.
    """

    _is_single_table = False

    def __init__(
        self,
        parent_table_name,
        child_table_name,
        child_foreign_key,
        conditional_column_name,
        conditional_values,
    ):
        super().__init__()
        _validate_foreign_to_primary_key_subset_input(
            parent_table_name,
            child_table_name,
            child_foreign_key,
            conditional_column_name,
            conditional_values,
        )
        self.parent_table_name = parent_table_name
        self.child_table_name = child_table_name
        self.child_foreign_key = child_foreign_key
        self.conditional_column_name = conditional_column_name
        self.conditional_values = conditional_values
        self._parent_primary_key = None

    def _get_scored_tables(self, metadata=None):
        return {self.child_table_name}

    def _validate_data(self, data, metadata=None):
        """Check that both tables and all the referenced columns exist in the data."""
        table_to_columns = {
            self.parent_table_name: [self.conditional_column_name],
            self.child_table_name: [self.child_foreign_key],
        }
        for table_name, table_columns in table_to_columns.items():
            if table_name not in data:
                raise ConstraintNotApplicableError(
                    f"The table '{table_name}' is missing from the data."
                )

            columns = data[table_name].columns
            missing_columns = [
                column_name for column_name in table_columns if column_name not in columns
            ]
            if missing_columns:
                missing_columns = "', '".join(missing_columns)
                raise ConstraintNotApplicableError(
                    f"The column(s) '{missing_columns}' are missing from the table '{table_name}'."
                )

        primary_key = _get_primary_key(metadata, self.parent_table_name)
        if primary_key not in data[self.parent_table_name].columns:
            raise ConstraintNotApplicableError(
                f"The column(s) '{primary_key}' are missing from the table "
                f"'{self.parent_table_name}'."
            )

        self._parent_primary_key = primary_key

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
        primary_key = self._parent_primary_key
        if primary_key is None:
            primary_key = _get_primary_key(metadata, self.parent_table_name)

        return _validate_foreign_to_primary_key_subset(
            data,
            primary_key,
            self.parent_table_name,
            self.child_table_name,
            self.child_foreign_key,
            self.conditional_column_name,
            self.conditional_values,
        )
