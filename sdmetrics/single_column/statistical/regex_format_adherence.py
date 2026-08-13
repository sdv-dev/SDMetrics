"""Regex Format Adherence Metric."""

import re
import warnings

import pandas as pd

from sdmetrics.goal import Goal
from sdmetrics.single_column.base import SingleColumnMetric


class RegexFormatAdherence(SingleColumnMetric):
    """Regex format adherence  metric.

    The proportion of data points in the synthetic data that match
    the given regex format.

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

    name = 'RegexFormatAdherence'
    goal = Goal.MAXIMIZE
    min_value = 0.0
    max_value = 1.0

    @staticmethod
    def _validate_regex_format(regex_format):
        if not isinstance(regex_format, str):
            raise ValueError('`regex_format` must be a string.')

        try:
            re.compile(regex_format)
        except re.error as exception:
            raise ValueError(f"Invalid regex format string '{regex_format}'.") from exception

    @staticmethod
    def _validate_regex_column(column, regex_format, compare=None):
        """Return values from the column that match the specified regex format.

        Args:
            column (pd.Series):
                Column to evaluate.
            regex_format (str):
                The regex format.
            compare (pd.Series | pd.DataFrame, optional):
                Dataframe containing groups to compare against.

        Returns:
            tuple (pd.Series, pd.DataFrame):
                * A series of values from the column that match the regex format.
                * A dataframe containing seperated regex groups.
        """
        regex_column = column.str.fullmatch(regex_format, na=False)

        groups = None
        if re.compile(regex_format).groups:
            groups = column[regex_column].str.extract(regex_format)
            if compare is not None and not compare.empty:
                valid_groups = set(map(tuple, compare.dropna().to_numpy()))
                regex_column = groups.apply(tuple, axis=1).isin(valid_groups)

        valid = pd.isna(column) | regex_column

        return column[valid], groups

    @classmethod
    def compute(cls, real_data, synthetic_data, regex_format):
        """Compute the regex format adherence metric.

        Args:
            real_data (pandas.Series):
                The real data.
            synthetic_data (pandas.Series):
                The synthetic data.
            regex_format (str):
                A string containing the regex format to check against.

        Returns:
            float:
                The proportion of data points in the synthetic data that match the regex format.
        """
        cls._validate_regex_format(regex_format)

        real_valid, real_groups = cls._validate_regex_column(real_data, regex_format)
        if len(real_valid) != len(real_data):
            invalid_values = real_data[~real_data.index.isin(real_valid.index)]
            num_examples = 2
            message = (
                'Some values in the real data do not match the specified regex format: '
                + ', '.join(f"'{value}'" for value in invalid_values.head(num_examples).astype(str))
            )

            remaining = len(invalid_values) - num_examples
            if remaining > 0:
                message += f' + {remaining} more.'

            warnings.warn(message)

        synthetic_valid, _ = cls._validate_regex_column(synthetic_data, regex_format, real_groups)
        score = len(synthetic_valid) / len(synthetic_data)

        return score
