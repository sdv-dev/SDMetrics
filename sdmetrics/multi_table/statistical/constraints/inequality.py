"""Inequality Constraint."""

import numpy as np
import pandas as pd
from pandas.api.types import is_object_dtype

from sdmetrics.multi_table.statistical.constraints._utils import (
    _get_is_valid_dict,
    _is_list_of_type,
    cast_to_datetime64,
    match_datetime_precision,
)
from sdmetrics.multi_table.statistical.constraints.base import BaseConstraint
from sdmetrics.multi_table.statistical.constraints.error import ConstraintNotApplicableError


class Inequality(BaseConstraint):
    """Constraint for inequality columns.

    Check that `high_column_name` is greater than `low_column_name`.

    Args:
        low_column_name (str):
            Name of the column that contains the low values.
        high_column_name (str):
            Name of the column that contains the high values.
        strict_boundaries (bool):
            Whether the comparison of the values should be strict ``>=`` or
            not ``>``. Defaults to False.
        table_name (str, optional):
            The name of the table that contains the columns. Optional if the
            data is only a single table. Defaults to None.
    """

    def __init__(self, low_column_name, high_column_name, strict_boundaries=False, table_name=None):
        super().__init__()
        if not isinstance(table_name, str):
            raise ValueError("The 'table_name' parameter must be a string.")

        if not _is_list_of_type([low_column_name, high_column_name], str):
            raise ValueError('`low_column_name` and `high_column_name` must be strings.')

        if not isinstance(strict_boundaries, bool):
            raise ValueError('`strict_boundaries` must be a boolean.')

        self._low_column_name = low_column_name
        self._high_column_name = high_column_name
        self._operator = np.greater if strict_boundaries else np.greater_equal
        self.table_name = table_name

    def _validate_data(self, data, metadata=None):
        """Check that the table and all the referenced columns exist in the data."""
        if self.table_name not in data:
            raise ConstraintNotApplicableError(
                f"The table '{self.table_name}' is missing from the data."
            )

        columns = data[self.table_name].columns
        range_columns = [self._low_column_name, self._high_column_name]
        missing_columns = [
            column_name for column_name in range_columns if column_name not in columns
        ]
        if missing_columns:
            missing_columns = "', '".join(missing_columns)
            raise ConstraintNotApplicableError(
                f"The column(s) '{missing_columns}' are missing from the table '{self.table_name}'."
            )

    def _get_data(self, data):
        low = data[self._low_column_name].to_numpy()
        high = data[self._high_column_name].to_numpy()
        return low, high

    def _get_is_datetime(self, metadata, table_name):
        return (
            metadata['tables'][table_name]['columns'][self._low_column_name]['sdtype'] == 'datetime'
        )

    def _get_datetime_format(self, metadata, table_name, column_name):
        datetime_format = metadata['tables'][table_name]['columns'][column_name].get(
            'datetime_format'
        )
        return datetime_format

    def _get_valid_table_data(self, table_data, metadata, table_name):
        low, high = self._get_data(table_data)
        is_datetime = self._get_is_datetime(metadata, table_name)
        if is_datetime and is_object_dtype(table_data[self._low_column_name]):
            low_format = self._get_datetime_format(metadata, table_name, self._low_column_name)
            high_format = self._get_datetime_format(metadata, table_name, self._high_column_name)
            low = cast_to_datetime64(low, low_format)
            high = cast_to_datetime64(high, high_format)

            format_matches = bool(low_format == high_format)
            if not format_matches:
                low, high = match_datetime_precision(
                    low=low,
                    high=high,
                    low_datetime_format=low_format,
                    high_datetime_format=high_format,
                )

        return pd.isna(low) | pd.isna(high) | self._operator(high, low)

    def _is_valid(self, data, metadata):
        """Check whether `high` is greater than `low` in each row.

        Args:
            data (dict[str, pd.DataFrame]):
                A dictionary mapping each table name to its data.
            metadata (dict):
               The multi table metadata.

        Returns:
            dict[str, pd.Series]:
                Whether each row is valid.
        """
        table_name = self._get_single_table_name(metadata)
        is_valid = _get_is_valid_dict(data, table_name)
        valid_table_rows = self._get_valid_table_data(data[table_name], metadata, table_name)
        is_valid[table_name] = pd.Series(valid_table_rows)

        return is_valid
