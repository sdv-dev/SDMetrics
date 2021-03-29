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

        This is done by grouping all the columns that are compatible with the
        underlying ColumnPairs metric in groups of 2 and then evaluating them
        using the ColumnPairs metric.

        The output is the average of the scores obtained.

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

    @classmethod
    def normalize(cls, raw_score):
        """Returns the `raw_score` as is, since it is already normalized.

        Args:
            raw_score (float):
                The value of the metric from `compute`.

        Returns:
            float:
                The normalized value of the metric
        """
        assert cls.min_value == 0.0
        return super().normalize(raw_score)


class ContinuousKLDivergence(MultiColumnPairsMetric):
    """MultiColumnPairsMetric based on ColumnPairs ContinuousKLDivergence.

    This approximates the KL divergence by binning the continuous values
    to turn them into categorical values and then computing the relative
    entropy. Afterwards normalizes the value applying ``1 / (1 + KLD)``.

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
            ColumnPairs ContinuousKLDivergence.
        field_types (dict):
            Field types to which the SingleColumn metric will be applied.
    """

    field_types = ('numerical', )
    column_pairs_metric = column_pairs.statistical.kl_divergence.ContinuousKLDivergence


class DiscreteKLDivergence(MultiColumnPairsMetric):
    """MultiColumnPairsMetric based on ColumnPairs DiscreteKLDivergence.

    This computes the KL divergence and afterwards normalizes the
    value applying ``1 / (1 + KLD)``.

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
            ColumnPairs DiscreteKLDivergence.
        field_types (dict):
            Field types to which the SingleColumn metric will be applied.
    """

    field_types = ('boolean', 'categorical')
    column_pairs_metric = column_pairs.statistical.kl_divergence.DiscreteKLDivergence
