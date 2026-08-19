"""Fixed Incerements Constraint."""

import pandas as pd

from sdmetrics.multi_table.statistical.constraints.base import BaseConstraint
from sdmetrics.multi_table.statistical.constraints.utils import _get_is_valid_dict

class FixedIncrements(BaseConstraint):

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
                not by the increment. The length of the Series
                will be equal to the length of the input column.
                The length of the dictionary will be equal to the
                number of tables in the data and contain the same
                table names.
        """
        table_name = self._get_single_table_name(metadata)
        is_valid = _get_is_valid_dict(data, table_name)
        valid = self._check_if_divisible(data, table_name, self.column_name, self.increment_value)
        is_valid[table_name] = valid
        return is_valid
