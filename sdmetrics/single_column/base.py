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
    """

    name = None
    goal = None
    min_value = None
    max_value = None

    @staticmethod
    def compute(real_data, synthetic_data):
        """Compute this metric.

        Args:
            real_data (Union[numpy.ndarray, pandas.Series]):
                The values from the real dataset, passed as a 1d numpy
                array or as a pandas.Series.
            synthetic_data (Union[numpy.ndarray, pandas.Series]):
                The values from the synthetic dataset, passed as a 1d numpy
                array or as a pandas.Series.

        Returns:
            Union[float, tuple[float]]:
                Metric output.
        """
        raise NotImplementedError()
