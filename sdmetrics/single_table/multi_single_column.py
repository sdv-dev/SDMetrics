"""SingleTable metrics based on applying a SingleColumnMetric on all the columns."""

import numpy as np

from sdmetrics import single_column
from sdmetrics.single_table.base import SingleTableMetric
from sdmetrics.utils import NestedAttrsMeta


class MultiColumnMetric(SingleTableMetric, metaclass=NestedAttrsMeta('single_column_metric')):
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
    """

    single_column_metric = None

    @classmethod
    def _dtype_match(cls, column):
        return any(
            column.dtype.kind == np.dtype(dtype).kind
            for dtype in cls.single_column_metric.dtypes
        )

    @classmethod
    def compute(cls, real_data, synthetic_data):
        """Compute this metric.

        Args:
            real_data (pandas.DataFrame):
                The values from the real dataset.
            synthetic_data (pandas.DataFrame):
                The values from the synthetic dataset.

        Returns:
            Union[float, tuple[float]]:
                Metric output.
        """
        if set(real_data.columns) != set(synthetic_data.columns):
            raise ValueError('`real_data` and `synthetic_data` must have the same columns')

        values = []
        for column_name, real_column in real_data.items():
            if cls._dtype_match(real_column):
                x1 = real_column.values
                x2 = synthetic_data[column_name].values

                values.append(cls.single_column_metric.compute(x1, x2))

        return np.mean(values)


class CSTest(MultiColumnMetric):
    """MultiColumnMetric based on SingleColumn CSTest."""

    single_column_metric = single_column.statistical.CSTest


class KSTest(MultiColumnMetric):
    """MultiColumnMetric based on SingleColumn KSTest."""

    single_column_metric = single_column.statistical.KSTest
