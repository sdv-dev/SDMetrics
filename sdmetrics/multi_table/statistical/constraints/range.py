"""Range Constraint."""

import pandas as pd
from pandas.api.types import is_object_dtype

from sdmetrics.multi_table.statistical.constraints.base import BaseConstraint
from sdmetrics.multi_table.statistical.constraints.utils import _get_is_valid_dict, cast_to_datetime64

class Range(BaseConstraint):

    def _get_valid_table_data(self, table_data):
        low = table_data[self._low_column_name]
        mid = table_data[self._middle_column_name]
        high = table_data[self._high_column_name]

        if self._is_datetime and is_object_dtype(self._dtype):
            low = cast_to_datetime64(low, self._low_datetime_format)
            mid = cast_to_datetime64(mid, self._middle_datetime_format)
            high = cast_to_datetime64(high, self._high_datetime_format)

        low_is_nan = pd.isna(low)
        mid_is_nan = pd.isna(mid)
        high_is_nan = pd.isna(high)

        low_lt_middle = low_is_nan | mid_is_nan | self._operator(low, mid)
        mid_lt_high = mid_is_nan | high_is_nan | self._operator(mid, high)
        low_lt_high = low_is_nan | high_is_nan | self._operator(low, high)

        return low_lt_middle & mid_lt_high & low_lt_high
    
    def _is_valid(self, data, metadata):
        """Check whether the `middle` column is between the `low` and `high` columns.

        Args:
            data (dict[str, pd.DataFrame]):
                Table data.

        Returns:
            dict[str, pd.Series]:
                Whether each row is valid.
        """
        table_name = self._get_single_table_name(metadata)
        is_valid = _get_is_valid_dict(data, table_name)
        is_valid[table_name] = self._get_valid_table_data(data[table_name])

        return is_valid