"""ColumnPair metric for Cardinality Boundary Adherence."""

import pandas as pd

from sdmetrics.column_pairs.base import ColumnPairsMetric
from sdmetrics.goal import Goal


class CardinalityBoundaryAdherence(ColumnPairsMetric):
    """Cardinality Boundary Adherence metric.

    Computes the percentage of synthetic parents whose cardinality
    falls within the min/max range of cardinality in the real data.

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

    name: 'CardinalityBoundaryAdherence'
    goal: Goal.MAXIMIZE
    min_value: 0.0
    max_value: 1.0

    @staticmethod
    def compute_breakdown(real_data, synthetic_data):
        """Calculate the percentage of synthetic parents with cardinality in the correct range.

        Args:
            real_data (tuple(pd.Series, pd.Series)):
                A tuple with the real primary key Series as the first element and real
                foreign keys Series as the second element.
            synthetic_data (tuple(pd.Series, pd.Series)):
                A tuple with the synthetic primary key as the first element and synthetic
                foreign keys as the second element.

        Returns:
            dict
                Metric output.
        """
        real_cardinality = pd.DataFrame(index=real_data[0].copy())
        real_cardinality['cardinality'] = real_data[1].value_counts()
        real_cardinality = real_cardinality.fillna(0)
        synthetic_cardinality = pd.DataFrame(index=synthetic_data[0].copy())
        synthetic_cardinality['cardinality'] = synthetic_data[1].value_counts()
        synthetic_cardinality = synthetic_cardinality.fillna(0)

        min_cardinality = real_cardinality['cardinality'].min()
        max_cardinality = real_cardinality['cardinality'].max()

        valid_cardinality = sum(
            synthetic_cardinality['cardinality'].between(min_cardinality, max_cardinality)
        )
        score = valid_cardinality / len(synthetic_cardinality)

        return {'score': score}

    @classmethod
    def compute(cls, real_data, synthetic_data):
        """Calculate the percentage of synthetic parents with cardinality in the correct range.

        Args:
            real_data (tuple(pd.Series, pd.Series)):
                A tuple with the real primary key Series as the first element and real
                foreign keys Series as the second element.
            synthetic_data (tuple(pd.Series, pd.Series)):
                A tuple with the synthetic primary key as the first element and synthetic
                foreign keys as the second element.

        Returns:
            float:
                Metric output.
        """
        return cls.compute_breakdown(real_data, synthetic_data)['score']
