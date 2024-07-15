"""SingleTable metrics based on applying a SingleColumnMetric on all the columns."""

import numpy as np

from sdmetrics import single_column
from sdmetrics.single_table.base import SingleTableMetric
from sdmetrics.utils import get_columns_from_metadata, nested_attrs_meta


class MultiSingleColumnMetric(
    SingleTableMetric, metaclass=nested_attrs_meta('single_column_metric')
):
    """SingleTableMetric subclass that applies a SingleColumnMetric on each column.

    This class can either be used by creating a subclass that inherits from it and
    sets the SingleColumn Metric as the ``single_column_metric`` attribute,
    or by creating an instance of this class passing the underlying SingleColumn
    metric as an argument.

    Attributes:
        name (str):
            Name to use when reports about this metric are printed.
        goal (sdmetrics.goal.Goal):
            The goal of this metric.
        min_value (Union[float, tuple[float]]):
            Minimum value or values that this metric can take.
        max_value (Union[float, tuple[float]]):
            Maximum value or values that this metric can take.
        single_column_metric (sdmetrics.single_column.base.SingleColumnMetric):
            SingleColumn metric to apply.
        field_types (dict):
            Field types to which the SingleColumn metric will be applied.
    """

    single_column_metric = None
    single_column_metric_kwargs = None
    field_types = None

    def __init__(self, single_column_metric=None, **single_column_metric_kwargs):
        self.single_column_metric = single_column_metric
        self.single_column_metric_kwargs = single_column_metric_kwargs
        self.compute = self._compute

    def _compute(self, real_data, synthetic_data, metadata=None, store_errors=False, **kwargs):
        """Compute this metric for all columns.

        This is done by computing the underlying SingleColumn metric to all the
        columns that are compatible with it.

        The output is a mapping of column name to the score of that column.

        Args:
            real_data (pandas.DataFrame):
                The values from the real dataset.
            synthetic_data (pandas.DataFrame):
                The values from the synthetic dataset.
            metadata (dict):
                Table metadata dict.
            store_errors (bool):
                Whether or not to store any metric computation errors in the results.
            **kwargs:
                Any additional keyword arguments will be passed down
                to the single column metric

        Returns:
            Dict[string -> Union[float, tuple[float]]]:
                A mapping of column name to metric output.
        """
        real_data, synthetic_data, metadata = self._validate_inputs(
            real_data, synthetic_data, metadata
        )

        fields = self._select_fields(metadata, self.field_types)
        invalid_cols = set(get_columns_from_metadata(metadata).keys()) - set(fields)

        scores = {col: {'score': np.nan} for col in invalid_cols}
        for column_name, real_column in real_data.items():
            if column_name in fields:
                real_column = real_column.to_numpy()
                synthetic_column = synthetic_data[column_name].to_numpy()

                try:
                    score = self.single_column_metric.compute_breakdown(
                        real_column,
                        synthetic_column,
                        **(self.single_column_metric_kwargs or {}),
                        **kwargs,
                    )
                    scores[column_name] = score
                except Exception as error:
                    if store_errors:
                        scores[column_name] = {'error': error}
                    else:
                        raise error

        return scores

    @classmethod
    def compute(cls, real_data, synthetic_data, metadata=None, **kwargs):
        """Compute this metric.

        This is done by computing the underlying SingleColumn metric to all the
        columns that are compatible with it.

        The output is the average of the scores obtained.

        Args:
            real_data (pandas.DataFrame):
                The values from the real dataset.
            synthetic_data (pandas.DataFrame):
                The values from the synthetic dataset.
            metadata (dict):
                Table metadata dict.
            **kwargs:
                Any additional keyword arguments will be passed down
                to the single column metric

        Returns:
            Union[float, tuple[float]]:
                Metric output.
        """
        scores = cls._compute(cls, real_data, synthetic_data, metadata, **kwargs)
        return np.nanmean([breakdown['score'] for breakdown in scores.values()])

    @classmethod
    def compute_breakdown(cls, real_data, synthetic_data, metadata=None, **kwargs):
        """Compute this metric broken down by column.

        This is done by computing the underlying SingleColumn metric to all the
        columns that are compatible with it.

        The output is a mapping of column to the column's score.

        Args:
            real_data (pandas.DataFrame):
                The values from the real dataset.
            synthetic_data (pandas.DataFrame):
                The values from the synthetic dataset.
            metadata (dict):
                Table metadata dict.
            **kwargs:
                Any additional keyword arguments will be passed down
                to the single column metric

        Returns:
            Dict[string -> Union[float, tuple[float]]]:
                A mapping of column name to metric output.
        """
        return cls._compute(cls, real_data, synthetic_data, metadata, store_errors=True, **kwargs)

    @classmethod
    def normalize(cls, raw_score):
        """Return the `raw_score` as is, since it is already normalized.

        Args:
            raw_score (float):
                The value of the metric from `compute`.

        Returns:
            float:
                The normalized value of the metric
        """
        assert cls.min_value == 0.0
        return super().normalize(raw_score)


class CSTest(MultiSingleColumnMetric):
    """MultiSingleColumnMetric based on SingleColumn CSTest.

    This function applies the single column ``CSTest`` metric to all
    the discrete columns found in the table and then returns the average
    of all the scores obtained.
    """

    field_types = ('boolean', 'categorical')
    single_column_metric = single_column.statistical.CSTest


class KSComplement(MultiSingleColumnMetric):
    """MultiSingleColumnMetric based on SingleColumn KSComplement.

    This function applies the single column ``KSComplement`` metric to all
    the numerical columns found in the table and then returns the average
    of all the scores obtained.
    """

    field_types = ('numerical', 'datetime')
    single_column_metric = single_column.statistical.KSComplement


class StatisticSimilarity(MultiSingleColumnMetric):
    """MultiSingleColumnMetric based on SingleColumn StatisticSimilarity.

    Apply the desired statistic to compare the real and synthetic data.
    """

    field_types = ('numerical', 'datetime')
    single_column_metric = single_column.statistical.StatisticSimilarity


class BoundaryAdherence(MultiSingleColumnMetric):
    """MultiSingleColumnMetric based on SingleColumn BoundaryAdherence.

    Compute the fraction of rows in the synthetic data that are within the min and max
    bounds of the real data.
    """

    field_types = ('numerical', 'datetime')
    single_column_metric = single_column.statistical.BoundaryAdherence


class MissingValueSimilarity(MultiSingleColumnMetric):
    """MultiSingleColumnMetric based on SingleColumn MissingValueSimilarity.

    Compare the percentage of missing values between the real and synthetic data.
    """

    field_types = ('numerical', 'datetime')
    single_column_metric = single_column.statistical.MissingValueSimilarity


class CategoryCoverage(MultiSingleColumnMetric):
    """MultiSingleColumnMetric based on SingleColumn CategoryCoverage.

    Compute the fraction of real data categories that are present in the synthetic data.
    """

    field_types = ('categorical', 'boolean')
    single_column_metric = single_column.statistical.CategoryCoverage


class TVComplement(MultiSingleColumnMetric):
    """MultiSingleColumnMetric based on SingleColumn TVComplement.

    Compute the complement of the total variaton distance between the real and synthetic data
    """

    field_types = ('categorical', 'boolean')
    single_column_metric = single_column.statistical.TVComplement


class RangeCoverage(MultiSingleColumnMetric):
    """MultiSingleColumnMetric based on SingleColumn RangeCoverage.

    Compute the complement of the total variaton distance between the real and synthetic data
    """

    field_types = ('numerical', 'datetime')
    single_column_metric = single_column.statistical.RangeCoverage
