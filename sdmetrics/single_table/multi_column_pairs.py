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
        field_types (dict):
            Field types to which the SingleColumn metric will be applied.
    """

    column_pairs_metric = None
    column_pairs_metric_kwargs = None
    field_types = None

    def __init__(self, column_pairs_metric, **column_pairs_metric_kwargs):
        self.column_pairs_metric = column_pairs_metric
        self.column_pairs_metric_kwargs = column_pairs_metric_kwargs
        self.compute = self._compute

    def _compute(self, real_data, synthetic_data, metadata=None, **kwargs):
        """Compute this metric.

        Args:
            real_data (Union[numpy.ndarray, pandas.DataFrame]):
                The values from the real dataset.
            synthetic_data (Union[numpy.ndarray, pandas.DataFrame]):
                The values from the synthetic dataset.
            metadata (dict):
                Table metadata dict.
            **kwargs:
                Any additional keyword arguments will be passed down
                to the column pairs metric

        Returns:
            Union[float, tuple[float]]:
                Metric output.
        """
        metadata = self._validate_inputs(real_data, synthetic_data, metadata)

        fields = self._select_fields(metadata, self.field_types)

        values = []
        for columns in combinations(fields, r=2):
            real = real_data[list(columns)]
            synthetic = synthetic_data[list(columns)]
            values.append(self.column_pairs_metric.compute(real, synthetic))

        return np.nanmean(values)

    @classmethod
    def compute(cls, real_data, synthetic_data, metadata=None, **kwargs):
        """Compute this metric.

        Args:
            real_data (Union[numpy.ndarray, pandas.DataFrame]):
                The values from the real dataset.
            synthetic_data (Union[numpy.ndarray, pandas.DataFrame]):
                The values from the synthetic dataset.
            metadata (dict):
                Table metadata dict.
            **kwargs:
                Any additional keyword arguments will be passed down
                to the column pairs metric

        Returns:
            Union[float, tuple[float]]:
                Metric output.
        """
        return cls._compute(cls, real_data, synthetic_data, metadata, **kwargs)


class ContinuousKLDivergence(MultiColumnPairsMetric):
    """MultiColumnPairsMetric based on ColumnPairs ContinuousKLDivergence."""

    field_types = ('numerical', )
    column_pairs_metric = column_pairs.statistical.kl_divergence.ContinuousKLDivergence


class DiscreteKLDivergence(MultiColumnPairsMetric):
    """MultiColumnPairsMetric based on ColumnPairs DiscreteKLDivergence."""

    field_types = ('boolean', 'categorical')
    column_pairs_metric = column_pairs.statistical.kl_divergence.DiscreteKLDivergence
