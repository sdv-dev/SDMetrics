"""InterRowMSAS module."""

import numpy as np
import pandas as pd

from sdmetrics.goal import Goal
from sdmetrics.single_column.statistical.kscomplement import KSComplement


class InterRowMSAS:
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
    def compute(real_data, synthetic_data, n_rows_diff=1, apply_log=False):
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
        real_keys, real_values = real_data
        synthetic_keys, synthetic_values = synthetic_data

        if apply_log:
            real_values = np.log(real_values)
            synthetic_values = np.log(synthetic_values)

        def calculate_differences(keys, values):
            differences = []
            for key in keys.unique():
                group_values = values[keys == key].to_numpy()
                if len(group_values) > n_rows_diff:
                    diff = group_values[n_rows_diff:] - group_values[:-n_rows_diff]
                    differences.append(np.mean(diff))
            return pd.Series(differences)

        real_diff = calculate_differences(real_keys, real_values)
        synthetic_diff = calculate_differences(synthetic_keys, synthetic_values)

        return KSComplement.compute(real_diff, synthetic_diff)
