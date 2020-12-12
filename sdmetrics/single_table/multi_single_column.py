"""SingleTable metrics based on applying a SingleColumnMetric on all the columns."""

import numpy as np
from rdt import HyperTransformer

from sdmetrics import single_column
from sdmetrics.single_table.base import SingleTableMetric
from sdmetrics.utils import NestedAttrsMeta


class MultiSingleColumnMetric(SingleTableMetric,
                              metaclass=NestedAttrsMeta('single_column_metric')):
    """SingleTableMetric subclass that applies a SingleColumnMetric on each column.

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
    field_types = None

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

        fields = cls._select_fields(metadata, cls.field_types)
        values = []
        for column_name, real_column in real_data.items():
            if column_name in fields:
                x1 = real_column.values
                x2 = synthetic_data[column_name].values

                values.append(cls.single_column_metric.compute(x1, x2))

        return np.nanmean(values)


class CSTest(MultiSingleColumnMetric):
    """MultiSingleColumnMetric based on SingleColumn CSTest."""

    field_types = ('boolean', 'categorical')
    single_column_metric = single_column.statistical.CSTest


class KSTest(MultiSingleColumnMetric):
    """MultiSingleColumnMetric based on SingleColumn KSTest."""

    field_types = ('numerical', )
    single_column_metric = single_column.statistical.KSTest


class KSTestExtended(MultiSingleColumnMetric):
    """KSTest variation that transforms everything to numerical before comparing."""

    single_column_metric = single_column.statistical.KSTest
    field_types = ('numerical', 'categorical', 'boolean', 'datetime')

    @classmethod
    def compute(cls, real_data, synthetic_data, metadata=None):
        metadata = cls._validate_inputs(real_data, synthetic_data, metadata)
        transformer = HyperTransformer()
        fields = cls._select_fields(metadata, cls.field_types)
        real_data = transformer.fit_transform(real_data[fields])
        synthetic_data = transformer.transform(synthetic_data[fields])

        values = []
        for column_name, real_column in real_data.items():
            x1 = real_column.values
            x2 = synthetic_data[column_name].values
            values.append(cls.single_column_metric.compute(x1, x2))

        return np.nanmean(values)
