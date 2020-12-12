"""SingleTable metrics based on applying a ColumnPairsMetrics on all the possible column pairs."""

from itertools import combinations

import numpy as np

from sdmetrics import column_pairs
from sdmetrics.single_table.base import SingleTableMetric
from sdmetrics.utils import NestedAttrsMeta


class MultiColumnPairsMetric(SingleTableMetric, metaclass=NestedAttrsMeta('column_pairs_metric')):
    """SingleTableMetric subclass that applies a ColumnPairsMetric on each possible column pair.

    Attributes:
        name (str):
            Name to use when reports about this metric are printed.
        goal (sdmetrics.goal.Goal):
            The goal of this metric.
        min_value (Union[float, tuple[float]]):
            Minimum value or values that this metric can take.
        max_value (Union[float, tuple[float]]):
            Maximum value or values that this metric can take.
        column_pairs_metric (sdmetrics.column_pairs.base.ColumnPairsMetric):
            ColumnPairsMetric to apply.
    """

    column_pairs_metric = None

    @classmethod
    def compute(cls, real_data, synthetic_data):
        """Compute this metric.

        Args:
            real_data (Union[numpy.ndarray, pandas.DataFrame]):
                The values from the real dataset.
            synthetic_data (Union[numpy.ndarray, pandas.DataFrame]):
                The values from the synthetic dataset.

        Returns:
            Union[float, tuple[float]]:
                Metric output.
        """
        if set(real_data.columns) != set(synthetic_data.columns):
            raise ValueError('`real_data` and `synthetic_data` must have the same columns')

        real_data = real_data.select_dtypes(cls.column_pairs_metric.dtypes)
        synthetic_data = synthetic_data.select_dtypes(cls.column_pairs_metric.dtypes)

        values = []
        for columns in combinations(real_data.columns, r=2):
            real = real_data[list(columns)]
            synthetic = synthetic_data[list(columns)]
            values.append(cls.column_pairs_metric.compute(real, synthetic))

        return np.nanmean(values)


class ContinuousKLDivergence(MultiColumnPairsMetric):
    """MultiColumnPairsMetric based on ColumnPairs ContinuousKLDivergence."""

    column_pairs_metric = column_pairs.statistical.kl_divergence.ContinuousKLDivergence


class DiscreteKLDivergence(MultiColumnPairsMetric):
    """MultiColumnPairsMetric based on ColumnPairs DiscreteKLDivergence."""

    column_pairs_metric = column_pairs.statistical.kl_divergence.DiscreteKLDivergence
