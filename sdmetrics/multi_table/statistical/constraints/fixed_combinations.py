"""Fixed Combinations Constraint."""

from sdmetrics.multi_table.statistical.constraints.base import BaseConstraint
from sdmetrics.multi_table.statistical.constraints.error import ConstraintNotApplicableError
from sdmetrics.multi_table.statistical.constraints.utils import _get_is_valid_dict, _is_list_of_type


class FixedCombinations(BaseConstraint):
    def __init__(self, column_names, table_name=None):
        super().__init__()

        if not isinstance(table_name, str):
            raise ValueError("The 'table_name' parameter must be a string.")

        if not _is_list_of_type(column_names, str):
            raise ValueError("The 'column_names' parameter must be a list of strings.")

        self.column_names = column_names
        self.table_name = table_name
        self._joint_column = '#'.join(self.column_names)
        self._combinations = None

    def _validate_data(self, data, metadata=None):
        """Check that the table and all the referenced columns exist in the data."""
        if self.table_name not in data:
            raise ConstraintNotApplicableError(
                f"The table '{self.table_name}' is missing from the data."
            )

        columns = data[self.table_name].columns
        missing_columns = [
            column_name for column_name in self.column_names if column_name not in columns
        ]
        if missing_columns:
            missing_columns = "', '".join(missing_columns)
            raise ConstraintNotApplicableError(
                f"The column(s) '{missing_columns}' are missing from the table '{self.table_name}'."
            )

    def _is_valid(self, data, metadata=None):
        """Determine whether the data matches the constraint."""
        table_name = self._get_single_table_name(metadata)
        is_valid = _get_is_valid_dict(data, table_name)
        merged = data[table_name].merge(
            self._combinations, how='left', on=self.column_names, indicator=self._joint_column
        )
        valid_data = merged[self._joint_column] == 'both'
        valid_data.index = data[table_name].index
        is_valid[table_name] = valid_data
        return is_valid
