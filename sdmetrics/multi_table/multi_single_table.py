"""MultiTable metrics based on applying SingleTable metrics on all the tables."""

import numpy as np

from sdmetrics import single_table
from sdmetrics.multi_table.base import MultiTableMetric
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

    def __init__(self, single_table_metric):
        self.single_table_metric = single_table_metric
        self.compute = self._compute

    def _compute(self, real_data, synthetic_data, metadata=None):
        """Compute this metric.

        Args:
            real_data (dict[str, pandas.DataFrame]):
                The tables from the real dataset.
            synthetic_data (dict[str, pandas.DataFrame]):
                The tables from the synthetic dataset.
            metadata (dict):
                Multi-table metadata dict. If not passed, it is build based on the
                real_data fields and dtypes.
            **kwargs:
                Any additional keyword arguments will be passed down
                to the single table metric

        Returns:
            Union[float, tuple[float]]:
                Metric output.
        """
        if set(real_data.keys()) != set(synthetic_data.keys()):
            raise ValueError('`real_data` and `synthetic_data` must have the same tables')

        values = []
        for table_name, real_table in real_data.items():
            synthetic_table = synthetic_data[table_name]
            table_meta = metadata['tables'][table_name] if metadata else None

            score = self.single_table_metric.compute(real_table, synthetic_table, table_meta)
            values.append(score)

        return np.nanmean(values)

    @classmethod
    def compute(cls, real_data, synthetic_data, metadata=None, **kwargs):
        """Compute this metric.

        Args:
            real_data (dict[str, pandas.DataFrame]):
                The tables from the real dataset.
            synthetic_data (dict[str, pandas.DataFrame]):
                The tables from the synthetic dataset.
            metadata (dict):
                Multi-table metadata dict. If not passed, it is build based on the
                real_data fields and dtypes.
            **kwargs:
                Any additional keyword arguments will be passed down
                to the single table metric

        Returns:
            Union[float, tuple[float]]:
                Metric output.
        """
        return cls._compute(cls, real_data, synthetic_data, metadata, **kwargs)


class CSTest(MultiSingleTableMetric):
    """MultiSingleTableMetric based on SingleTable CSTest."""

    single_table_metric = single_table.multi_single_column.CSTest


class KSTest(MultiSingleTableMetric):
    """MultiSingleTableMetric based on SingleTable KSTest."""

    single_table_metric = single_table.multi_single_column.KSTest


class KSTestExtended(MultiSingleTableMetric):
    """MultiSingleTableMetric based on SingleTable KSTestExtended."""

    single_table_metric = single_table.multi_single_column.KSTestExtended


class LogisticDetection(MultiSingleTableMetric):
    """MultiSingleTableMetric based on SingleTable LogisticDetection."""

    single_table_metric = single_table.detection.LogisticDetection


class SVCDetection(MultiSingleTableMetric):
    """MultiSingleTableMetric based on SingleTable SVCDetection."""

    single_table_metric = single_table.detection.SVCDetection


class BNLikelihood(MultiSingleTableMetric):
    """MultiSingleTableMetric based on SingleTable BNLikelihood."""

    single_table_metric = single_table.bayesian_network.BNLikelihood


class BNLogLikelihood(MultiSingleTableMetric):
    """MultiSingleTableMetric based on SingleTable BNLogLikelihood."""

    single_table_metric = single_table.bayesian_network.BNLogLikelihood
