"""Range Constraint."""

import operator

import pandas as pd
from pandas.api.types import is_object_dtype

from sdmetrics.multi_table.statistical.constraints.base import BaseConstraint
from sdmetrics.multi_table.statistical.constraints.error import ConstraintNotApplicableError
from sdmetrics.multi_table.statistical.constraints._utils import (
    _get_is_valid_dict,
    _is_list_of_type,
    cast_to_datetime64,
)


class Range(BaseConstraint):
    """Constraint for range columns.

    Check that `middle_column_name` is between `low_column_name` and `high_column_name`.

    Args:
        low_column_name (str):
            Name of the column which will be the lower bound.
        middle_column_name (str):
            Name of the column that has to be between the lower bound and upper bound.
        high_column_name (str):
            Name of the column which will be the higher bound.
        strict_boundaries (bool, optional):
            Whether the comparison of the values should be strict `>=` or
            not `>` when comparing them.
            Defaults to True.
        table_name (str, optional):
            The name of the table that contains the columns. Optional if the
            data is only a single table. Defaults to None.
    """

    def __init__(
        self,
        low_column_name,
        middle_column_name,
        high_column_name,
        strict_boundaries,
        table_name=None,
    ):
        super().__init__()
        if not isinstance(table_name, str):
            raise ValueError("The 'table_name' parameter must be a string.")

        if not _is_list_of_type([low_column_name, middle_column_name, high_column_name], str):
            raise ValueError(
                '`low_column_name`, `middle_column_name` and `high_column_name` must be strings.'
            )

        if not isinstance(strict_boundaries, bool):
            raise ValueError('`strict_boundaries` must be a boolean.')

        self._low_column_name = low_column_name
        self._middle_column_name = middle_column_name
        self._high_column_name = high_column_name
        self._operator = operator.lt if strict_boundaries else operator.le
        self.table_name = table_name

    def _validate_data(self, data, metadata=None):
        """Check that the table and all the referenced columns exist in the data."""
        if self.table_name not in data:
            raise ConstraintNotApplicableError(
                f"The table '{self.table_name}' is missing from the data."
            )

        columns = data[self.table_name].columns
        range_columns = [self._low_column_name, self._middle_column_name, self._high_column_name]
        missing_columns = [
            column_name for column_name in range_columns if column_name not in columns
        ]
        if missing_columns:
            missing_columns = "', '".join(missing_columns)
            raise ConstraintNotApplicableError(
                f"The column(s) '{missing_columns}' are missing from the table '{self.table_name}'."
            )

    def _get_is_datetime(self, metadata, table_name):
        return (
            metadata['tables'][table_name]['columns'][self._low_column_name]['sdtype'] == 'datetime'
        )

    def _get_datetime_format(self, metadata, table_name, column_name):
        return metadata['tables'][table_name]['columns'][column_name].get('datetime_format')

    def _get_valid_table_data(self, table_data):
        low = table_data[self._low_column_name]
        mid = table_data[self._middle_column_name]
        high = table_data[self._high_column_name]

        _dtype = table_data[self._high_column_name].dtypes
        if self._is_datetime and is_object_dtype(_dtype):
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
            metadata (dict):
                Metadata as a dictionary.

        Returns:
            dict[str, pd.Series]:
                Whether each row is valid.
        """
        table_name = self._get_single_table_name(metadata)
        self._is_datetime = self._get_is_datetime(metadata, table_name)
        if self._is_datetime:
            self._low_datetime_format = self._get_datetime_format(
                metadata, table_name, self._low_column_name
            )
            self._middle_datetime_format = self._get_datetime_format(
                metadata, table_name, self._middle_column_name
            )
            self._high_datetime_format = self._get_datetime_format(
                metadata, table_name, self._high_column_name
            )

        is_valid = _get_is_valid_dict(data, table_name)
        is_valid[table_name] = self._get_valid_table_data(data[table_name])

        return is_valid
