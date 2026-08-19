"""Fixed Combinations Constraint."""

from sdmetrics.multi_table.statistical.constraints.base import BaseConstraint
from sdmetrics.multi_table.statistical.constraints.utils import _get_is_valid_dict

class FixedCombinations(BaseConstraint):
    def __init__(self, column_names, table_name=None):
        super().__init__()

        self.column_names = column_names
        self.table_name = table_name
        self._joint_column = '#'.join(self.column_names)
        self._combinations = None

    def _is_valid(self, data, metadata):
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
