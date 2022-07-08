"""SingleTable metrics based on applying a SingleColumnMetric on all the columns."""

import numpy as np
from rdt import HyperTransformer

from sdmetrics import single_column
from sdmetrics.single_table.base import SingleTableMetric
from sdmetrics.utils import nested_attrs_meta


class MultiSingleColumnMetric(SingleTableMetric,
                              metaclass=nested_attrs_meta('single_column_metric')):
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
        metadata = self._validate_inputs(real_data, synthetic_data, metadata)

        fields = self._select_fields(metadata, self.field_types)
        invalid_cols = set(metadata['fields'].keys()) - set(fields)

        scores = {col: np.nan for col in invalid_cols}
        for column_name, real_column in real_data.items():
            if column_name in fields:
                real_column = real_column.to_numpy()
                synthetic_column = synthetic_data[column_name].to_numpy()

                try:
                    score = self.single_column_metric.compute(
                        real_column,
                        synthetic_column,
                        **(self.single_column_metric_kwargs or {}),
                        **kwargs
                    )
                    scores[column_name] = score
                except Exception as error:
                    if store_errors:
                        scores[column_name] = error
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
        return np.nanmean(list(scores.values()))

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
        return cls._compute(
            cls, real_data, synthetic_data, metadata=None, store_errors=True, **kwargs)

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

    field_types = ('numerical', )
    single_column_metric = single_column.statistical.KSComplement


class KSTestExtended(MultiSingleColumnMetric):
    """KSComplement variation that transforms everything to numerical before comparing the tables.

    This is done by applying an ``rdt.HyperTransformer`` to the data with the
    default values and afterwards applying a regular single_column ``KSComplement``
    metric to all the generated numerical columns.
    """

    single_column_metric = single_column.statistical.KSComplement
    field_types = ('numerical', 'categorical', 'boolean', 'datetime')

    @classmethod
    def compute(cls, real_data, synthetic_data, metadata=None):
        """Compute this metric.

        Args:
            real_data (pandas.DataFrame):
                The values from the real dataset.
            synthetic_data (pandas.DataFrame):
                The values from the synthetic dataset.
            metadata (dict):
                Table metadata dict.

        Returns:
            Union[float, tuple[float]]:
                Metric output.
        """
        metadata = cls._validate_inputs(real_data, synthetic_data, metadata)
        transformer = HyperTransformer()
        fields = cls._select_fields(metadata, cls.field_types)
        transformer.detect_initial_config(real_data)
        real_data = transformer.fit_transform(real_data[fields])
        synthetic_data = transformer.transform(synthetic_data[fields])

        values = []
        for column_name, real_column in real_data.items():
            real_column = real_column.to_numpy()
            synthetic_column = synthetic_data[column_name].to_numpy()

            score = cls.single_column_metric.compute(real_column, synthetic_column)
            values.append(score)

        return np.nanmean(values)
