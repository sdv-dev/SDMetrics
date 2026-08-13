"""Datetime Format Adherence Metric."""

import re
import warnings
from datetime import datetime

import numpy as np
import pandas as pd
from pandas.api.types import is_string_dtype

from sdmetrics.goal import Goal
from sdmetrics.single_column.base import SingleColumnMetric


class DatetimeFormatAdherence(SingleColumnMetric):
    """Datetime format adherence  metric.

    The proportion of data points in the synthetic data that match
    the given datetime format.

    Attributes:
        name (str):
            Name to use when reports about this metric are printed.
        goal (sdmetrics.goal.Goal):
            The goal of this metric.
        min_value (Union[float, tuple[float]]):
            Minimum value or values that this metric can take.
        max_value (Union[float, tuple[float]]):
            Maximum value or values that this metric can take.
    """

    name = 'DatetimeFormatAdherence'
    goal = Goal.MAXIMIZE
    min_value = 0.0
    max_value = 1.0

    @staticmethod
    def _validate_datetime_format(datetime_format):
        if not isinstance(datetime_format, str):
            raise ValueError('`datetime_format` must be a string.')

        message = f"Invalid datetime format string '{datetime_format}'."
        try:
            formated_date = datetime.now().strftime(datetime_format)
        except Exception as exception:
            raise ValueError(message) from exception

        matches = re.findall('(%.)|(%)', formated_date)
        if matches:
            raise ValueError(message)

    @staticmethod
    def _filter_valid_datetime_rows(column, datetime_format):
        """Return values from the column that match the specified datetime format.

        Args:
            column (pd.Series):
                Column to evaluate. It must contain pd.Timestamp/string/datetime values.
                A column can contain multiple timezones, a single timezone, or no timezone.
                The column can be object, string or datetime64[ns] dtype.
            datetime_format (str):
                The datetime format.

        Returns:
            set:
                A set of values from the column that match the datetime format.
        """
        pandas_datetime_format = datetime_format.replace('%-', '%')
        utc = '%z' in datetime_format.lower()

        datetime_column = pd.to_datetime(
            column, errors='coerce', format=pandas_datetime_format, utc=utc
        )

        valid = pd.isna(column) | ~pd.isna(datetime_column)

        return column[valid]

    @classmethod
    def compute(cls, real_data, synthetic_data, datetime_format):
        """Compute the datetime adherence format metric.

        Args:
            real_data (pandas.Series):
                The real data.
            synthetic_data (pandas.Series):
                The synthetic data.
            datetime_format (str):
                A string containing the datetime format to check against.

        Returns:
            float:
                The proportion of data points in the synthetic data that match the datetime format.
        """
        real_data_nan = pd.isna(real_data)
        synthetic_data_nan = pd.isna(synthetic_data)
        if not is_string_dtype(synthetic_data[~synthetic_data_nan]):
            return np.nan

        cls._validate_datetime_format(datetime_format)
        if is_string_dtype(real_data[~real_data_nan]):
            real_valid_rows = cls._filter_valid_datetime_rows(real_data, datetime_format)
            if len(real_valid_rows) != len(real_data):
                warnings.warn('The real data does not match the given datetime format.')

        synthetic_valid_rows = cls._filter_valid_datetime_rows(synthetic_data, datetime_format)
        score = len(synthetic_valid_rows) / len(synthetic_data)

        return score
