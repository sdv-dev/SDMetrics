"""SequenceLengthSimilarity module."""

import pandas as pd

from sdmetrics.goal import Goal
from sdmetrics.single_column.base import SingleColumnMetric
from sdmetrics.single_column.statistical.kscomplement import KSComplement


class SequenceLengthSimilarity(SingleColumnMetric):
    """Sequence Length Similarity metric.

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

    name = 'Sequence Length Similarity'
    goal = Goal.MAXIMIZE
    min_value = 0.0
    max_value = 1.0

    @staticmethod
    def compute(real_data: pd.Series, synthetic_data: pd.Series) -> float:
        """Compute this metric.

        The length of a sequence is determined by the number of times the same sequence key occurs.
        For example if id_09231 appeared 150 times in the sequence key, then the sequence is of
        length 150. This metric compares the lengths of all sequence keys in the
        real data vs. the synthetic data.

        It works as follows:
            - Calculate the length of each sequence in the real data
            - Calculate the length of each sequence in the synthetic data
            - Apply the KSComplement metric to compare the similarities of the distributions
            - Return this score

        Args:
            real_data (pd.Series):
                The values from the real dataset.
            synthetic_data (pd.Series):
                The values from the synthetic dataset.

        Returns:
            float:
                The score.
        """
        return KSComplement.compute(real_data.value_counts(), synthetic_data.value_counts())
