"""Chained Inequality Constraint."""

import operator

import pandas as pd

from sdmetrics.multi_table.statistical.constraints.base import BaseConstraint
from sdmetrics.multi_table.statistical.constraints.error import ConstraintNotApplicableError
from sdmetrics.multi_table.statistical.constraints.utils import (
    _get_is_valid_dict,
    _is_datetime_type,
    _is_list_of_type,
    cast_to_datetime64,
)


class ChainedInequality(BaseConstraint):
    """Constraint for chained inequality across columns.

    This constraint is used to ensure that the values of a set of columns are
    monotonically increasing.

    Args:
        column_names (list[str]):
            A list of strings that represent the column names.
            They should appear in ascending order (lowest to highest).
        strict_boundaries (bool, optional):
            Whether a column must be strictly greater than previous one.
        table_name (str, optional):
            The name of the table that contains the columns. Optional if the
            data is only a single table. Defaults to None.
    """

    def __init__(self, column_names, strict_boundaries=True, table_name=None):
        super().__init__()
        if not isinstance(table_name, str):
            raise ValueError("The 'table_name' parameter must be a string.")

        if not _is_list_of_type(column_names, str):
            raise ValueError("The 'column_names' parameter must be a list of strings.")

        if not isinstance(strict_boundaries, bool):
            raise ValueError('`strict_boundaries` must be a boolean.')

        self._operator = operator.lt if strict_boundaries else operator.le
        self.column_names = column_names
        self.table_name = table_name

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

    def _check_chained(self, table_data, table_name, metadata):
        data = table_data[self.column_names].copy()
        table_metadata = metadata['tables'][table_name]

        is_datetime = table_metadata['columns'][self.column_names[0]]['sdtype'] == 'datetime'
        if is_datetime:
            for column_name in self.column_names:
                if not _is_datetime_type(data[column_name]):
                    continue

                datetime_format = table_metadata['columns'][column_name].get('datetime_format')
                data.loc[:, column_name] = cast_to_datetime64(
                    data[column_name], datetime_format=datetime_format
                )

        valid_rows = pd.Series(True, index=data.index)
        low_column = data[self.column_names[0]]
        for idx in range(1, len(self.column_names)):
            high_column = data[self.column_names[idx]]
            iteration_validity = (
                pd.isna(low_column) | pd.isna(high_column) | self._operator(low_column, high_column)
            )

            valid_rows = valid_rows & iteration_validity
            low_column = high_column.mask(pd.isna(high_column), low_column)
        return valid_rows

    def _is_valid(self, data, metadata=None):
        """Check that the data respects the chained inequalities, according to the column_names.

        Args:
            data (dict[str, pd.DataFrame]):
                The data dictionary.

        Returns:
            dict[str, pd.Series]:
                For the specified table and column, returns a Series
                which specifies if that row respect the inequality
                constraints with the chaining
        """
        metadata = metadata or self.metadata
        table_name = self._get_single_table_name(metadata)
        table_data = data[table_name]
        is_valid = _get_is_valid_dict(data, table_name)
        is_valid[table_name] = self._check_chained(table_data, table_name, metadata)
        return is_valid
