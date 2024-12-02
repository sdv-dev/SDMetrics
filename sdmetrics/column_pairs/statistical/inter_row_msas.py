"""InterRowMSAS module."""

import warnings

import numpy as np
import pandas as pd

from sdmetrics.column_pairs.base import ColumnPairsMetric
from sdmetrics.goal import Goal
from sdmetrics.single_column.statistical.kscomplement import KSComplement


class InterRowMSAS(ColumnPairsMetric):
    """Inter-Row Multi-Sequence Aggregate Similarity (MSAS) metric.

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

    name = 'Inter-Row Multi-Sequence Aggregate Similarity'
    goal = Goal.MAXIMIZE
    min_value = 0.0
    max_value = 1.0

    @staticmethod
    def _validate_inputs(real_data, synthetic_data, n_rows_diff, apply_log):
        for data in [real_data, synthetic_data]:
            if (
                not isinstance(data, tuple)
                or len(data) != 2
                or (not (isinstance(data[0], pd.Series) and isinstance(data[1], pd.Series)))
            ):
                raise ValueError('The data must be a tuple of two pandas series.')

        if not isinstance(n_rows_diff, int) or n_rows_diff < 1:
            raise ValueError("'n_rows_diff' must be an integer greater than zero.")

        if not isinstance(apply_log, bool):
            raise ValueError("'apply_log' must be a boolean.")

    @staticmethod
    def _apply_log(real_values, synthetic_values, apply_log):
        if apply_log:
            if pd.api.types.is_datetime64_any_dtype(
                real_values
            ) or pd.api.types.is_datetime64_any_dtype(synthetic_values):
                raise TypeError(
                    'Cannot compute log for datetime columns. '
                    "Please set 'apply_log' to False to use this metric."
                )

            num_invalid = sum(x <= 0 for x in pd.concat((real_values, synthetic_values)))
            if num_invalid:
                warnings.warn(
                    f'There are {num_invalid} non-positive values in your data, which cannot be '
                    "used with log. Consider changing 'apply_log' to False for a better result."
                )
            with warnings.catch_warnings():
                warnings.filterwarnings('ignore', message='.*encountered in log')
                real_values = np.log(real_values)
                synthetic_values = np.log(synthetic_values)

        return real_values, synthetic_values

    @staticmethod
    def _calculate_differences(keys, values, n_rows_diff, data_name):
        grouped = values.groupby(keys)
        group_sizes = grouped.size()

        num_invalid_groups = len(group_sizes[group_sizes <= n_rows_diff])
        if num_invalid_groups > 0:
            warnings.warn(
                f"n_rows_diff '{n_rows_diff}' is greater or equal to the "
                f'size of {num_invalid_groups} sequence keys in {data_name}.'
            )

        def diff_func(group):
            if len(group) <= n_rows_diff:
                return np.nan
            group = group.to_numpy()
            with warnings.catch_warnings():
                warnings.filterwarnings('ignore', message='Mean of empty slice')
                return np.nanmean(group[n_rows_diff:] - group[:-n_rows_diff])

        with warnings.catch_warnings():
            warnings.filterwarnings('ignore', message='invalid value encountered in.*')
            return grouped.apply(diff_func)

    @classmethod
    def compute(cls, real_data, synthetic_data, n_rows_diff=1, apply_log=False):
        """Compute this metric.

        This metric compares the inter-row differences of sequences in the real data
        vs. the synthetic data.

        It works as follows:
            - Calculate the difference between row r and row r+x for each row in the real data
            - Take the average over each sequence to form a distribution D_r
            - Do the same for the synthetic data to form a new distribution D_s
            - Apply the KSComplement metric to compare the similarities of (D_r, D_s)
            - Return this score

        Args:
            real_data (tuple[pd.Series, pd.Series]):
                A tuple of 2 pandas.Series objects. The first represents the sequence key
                of the real data and the second represents a continuous column of data.
            synthetic_data (tuple[pd.Series, pd.Series]):
                A tuple of 2 pandas.Series objects. The first represents the sequence key
                of the synthetic data and the second represents a continuous column of data.
            n_rows_diff (int):
                An integer representing the number of rows to consider when taking the difference.
            apply_log (bool):
                Whether to apply a natural log before taking the difference.

        Returns:
            float:
                The similarity score between the real and synthetic data distributions.
        """
        cls._validate_inputs(real_data, synthetic_data, n_rows_diff, apply_log)
        real_keys, real_values = real_data
        synthetic_keys, synthetic_values = synthetic_data
        real_values, synthetic_values = cls._apply_log(real_values, synthetic_values, apply_log)

        real_diff = cls._calculate_differences(real_keys, real_values, n_rows_diff, 'real_data')
        synthetic_diff = cls._calculate_differences(
            synthetic_keys, synthetic_values, n_rows_diff, 'synthetic_data'
        )

        return KSComplement.compute(real_diff, synthetic_diff)
