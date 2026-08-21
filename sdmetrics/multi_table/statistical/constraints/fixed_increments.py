"""Fixed Incerements Constraint."""

import pandas as pd

from sdmetrics.multi_table.statistical.constraints.base import BaseConstraint
from sdmetrics.multi_table.statistical.constraints.error import ConstraintNotApplicableError
from sdmetrics.multi_table.statistical.constraints.utils import _get_is_valid_dict


class FixedIncrements(BaseConstraint):
    """Contraint to check every value in a column is a multiple of the specified increment.

    Args:
        column_name (str):
            Name of the column.
        increment_value (int):
            The increment that each value in the column must be a multiple of. Must be greater
            than 0 and a whole number.
        table_name (str, optional):
            The name of the table that contains the column. Optional if the
            data is only a single table. Defaults to None.
    """

    def __init__(self, column_name, increment_value, table_name=None):
        super().__init__()

        if not isinstance(table_name, str):
            raise ValueError("The 'table_name' parameter must be a string.")
        if not isinstance(column_name, str):
            raise ValueError('`column_name` must be a string.')
        if not isinstance(increment_value, (int, float)):
            raise ValueError('`increment_value` must be an integer or float.')
        if table_name and not isinstance(table_name, str):
            raise ValueError('`table_name` must be a string if not None.')

        if increment_value <= 0:
            raise ValueError('`increment_value` must be greater than 0.')
        if increment_value % 1 != 0:
            raise ValueError('`increment_value` must be a whole number.')

        self.column_name = column_name
        self.table_name = table_name
        self.increment_value = increment_value

    def _validate_data(self, data, metadata):
        if self.table_name not in data:
            raise ConstraintNotApplicableError(
                f"The table '{self.table_name}' is missing from the data."
            )

        columns = data[self.table_name].columns
        if self.column_name not in columns:
            raise ConstraintNotApplicableError(
                f"The column '{self.column_name}' is missing from the table '{self.table_name}'."
            )

        col_sdtype = metadata['tables'][self.table_name]['columns'][self.column_name]['sdtype']
        if col_sdtype != 'numerical':
            raise ConstraintNotApplicableError(
                f"Column '{self.column_name}' has an incompatible sdtype ('{col_sdtype}')."
                " The column sdtype must be 'numerical'."
            )

    def _check_if_divisible(self, data, table_name, column_name, increment_value):
        """Check if a column is divisible by a given increment value.

        Args:
            data (dict[pd.DataFrame]):
                The data.

            table_name (str):
                Name of the table.

            column_name (str):
                Name of the table to check divisibility.

            increment_value (int):
                the number with which divisibility needs to be checked.
        """
        isnan = pd.isna(data[table_name][column_name])
        is_divisible = data[table_name][column_name] % increment_value == 0
        return isnan | is_divisible

    def _is_valid(self, data, metadata):
        """Determine if the data is evenly divisible by the increment.

        Args:
            data (dict[pd.DataFrame]):
                The data.

        Returns:
            (dict[pd.DataFrame]):
                For the specified table and column, returns a Series
                which specifies if that row is evenly divisible or
                not by the increment.
        """
        table_name = self._get_single_table_name(metadata)
        is_valid = _get_is_valid_dict(data, table_name)
        valid = self._check_if_divisible(data, table_name, self.column_name, self.increment_value)
        is_valid[table_name] = valid
        return is_valid
