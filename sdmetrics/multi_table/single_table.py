"""MultiTable metrics based on applying SingleTable metrics on all the tables."""

import numpy as np

from sdmetrics.multi_table.base import MultiTableMetric
from sdmetrics.single_table import detection, single_column
from sdmetrics.utils import NestedAttrsMeta


class MultiSingleTableMetric(MultiTableMetric, metaclass=NestedAttrsMeta('single_table_metric')):
    """MultiTableMetric subclass that applies a SingleTableMetric on each table.

    Attributes:
        name (str):
            Name to use when reports about this metric are printed.
        goal (sdmetrics.goal.Goal):
            The goal of this metric.
        min_value (Union[float, tuple[float]]):
            Minimum value or values that this metric can take.
        max_value (Union[float, tuple[float]]):
            Maximum value or values that this metric can take.
        single_table_metric (sdmetrics.single_table.base.SingleTableMetric):
            SingleTableMetric to apply.
    """

    single_table_metric = None

    @classmethod
    def compute(cls, real_data, synthetic_data):
        """Compute this metric.

        Args:
            real_data (dict[str, pandas.DataFrame]):
                The tables from the real dataset.
            synthetic_data (dict[str, pandas.DataFrame]):
                The tables from the synthetic dataset.

        Returns:
            Union[float, tuple[float]]:
                Metric output.
        """
        if set(real_data.keys()) != set(synthetic_data.keys()):
            raise ValueError('`real_data` and `synthetic_data` must have the same tables')

        values = []
        for table_name, real_table in real_data.items():
            synthetic_table = synthetic_data[table_name]
            values.append(cls.single_table_metric.compute(real_table, synthetic_table))

        return np.mean(values)


class CSTest(MultiSingleTableMetric):
    """MultiSingleTableMetric based on SingleTable CSTest."""

    single_table_metric = single_column.CSTest


class KSTest(MultiSingleTableMetric):
    """MultiSingleTableMetric based on SingleTable KSTest."""

    single_table_metric = single_column.KSTest


class LogisticDetection(MultiSingleTableMetric):
    """MultiSingleTableMetric based on SingleTable LogisticDetection."""

    single_table_metric = detection.LogisticDetection


class SVCDetection(MultiSingleTableMetric):
    """MultiSingleTableMetric based on SingleTable SVCDetection."""

    single_table_metric = detection.SVCDetection
