"""StatisticMSAS module."""

import pandas as pd

from sdmetrics.column_pairs.base import ColumnPairsMetric
from sdmetrics.goal import Goal
from sdmetrics.single_column.statistical.kscomplement import KSComplement


class StatisticMSAS(ColumnPairsMetric):
    """Statistic Multi-Sequence Aggregate Similarity (MSAS) metric.

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

    name = 'Statistic Multi-Sequence Aggregate Similarity'
    goal = Goal.MAXIMIZE
    min_value = 0.0
    max_value = 1.0

    @staticmethod
    def compute(real_data, synthetic_data, statistic='mean'):
        """Compute this metric.

        This metric compares the distribution of a given statistic across sequences
        in the real data vs. the synthetic data.

        It works as follows:
            - Calculate the specified statistic for each sequence in the real data
            - Form a distribution D_r from these statistics
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
            statistic (str):
                A string representing the statistic function to use when computing MSAS.

                Available options are:
                    - 'mean': The arithmetic mean of the sequence
                    - 'median': The median value of the sequence
                    - 'std': The standard deviation of the sequence
                    - 'min': The minimum value in the sequence
                    - 'max': The maximum value in the sequence

        Returns:
            float:
                The similarity score between the real and synthetic data distributions.
        """
        valid_statistics = ['mean', 'median', 'std', 'min', 'max']
        if statistic not in valid_statistics:
            raise ValueError(f'Invalid statistic: {statistic}. Choose from {valid_statistics}.')

        for data in [real_data, synthetic_data]:
            if (
                not isinstance(data, tuple)
                or len(data) != 2
                or (not (isinstance(data[0], pd.Series) and isinstance(data[1], pd.Series)))
            ):
                raise ValueError('The data must be a tuple of two pandas series.')

        real_keys, real_values = real_data
        synthetic_keys, synthetic_values = synthetic_data

        def calculate_statistics(keys, values):
            df = pd.DataFrame({'keys': keys, 'values': values})
            return df.groupby('keys')['values'].agg(statistic)

        real_stats = calculate_statistics(real_keys, real_values)
        synthetic_stats = calculate_statistics(synthetic_keys, synthetic_values)

        return KSComplement.compute(real_stats, synthetic_stats)
