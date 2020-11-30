"""Base SingleColumnMetric class."""

from sdmetrics.base import BaseMetric


class SingleColumnMetric(BaseMetric):
    """Base class for metrics that apply to individual columns.

    Attributes:
        name (str):
            Name to use when reports about this metric are printed.
        goal (sdmetrics.goal.Goal):
            The goal of this metric.
        min_value (Union[float, tuple[float]]):
            Minimum value or values that this metric can take.
        max_value (Union[float, tuple[float]]):
            Maximum value or values that this metric can take.
        dtypes (tuple[str]):
            The data types which this metric works on (i.e. ``('float', 'str')``).
    """

    name = None
    goal = None
    min_value = None
    max_value = None
    dtypes = ()

    @staticmethod
    def compute(real_data, synthetic_data):
        """Compute this metric.

        Args:
            real_data (Union[numpy.ndarray, pandas.Series]):
                The values from the real dataset.
            synthetic_data (Union[numpy.ndarray, pandas.Series]):
                The values from the synthetic dataset.

        Returns:
            Union[float, tuple[float]]:
                Metric output.
        """
        raise NotImplementedError()
