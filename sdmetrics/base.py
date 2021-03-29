"""BaseMetric class."""
import numpy as np

from sdmetrics.goal import Goal


class BaseMetric:
    """Base class for all the metrics in SDMetrics.

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

    @classmethod
    def get_subclasses(cls, include_parents=False):
        """Recursively find subclasses of this metric.

        If ``include_parents`` is passed as ``True``, intermediate child classes
        that also have subclasses will be included. Otherwise, only classes
        without subclasses will be included to ensure that they are final
        implementations and are ready to be run on data.

        Args:
            include_parents (bool):
                Whether to include subclasses which are parents to
                other classes. Defaults to ``False``.
        """
        subclasses = dict()
        for child in cls.__subclasses__():
            grandchildren = child.get_subclasses(include_parents)
            subclasses.update(grandchildren)
            if include_parents or not grandchildren:
                subclasses[child.__name__] = child

        return subclasses

    @staticmethod
    def compute(real_data, synthetic_data):
        """Compute this metric.

        Args:
            real_data:
                The values from the real dataset.
            synthetic_data:
                The values from the synthetic dataset.

        Returns:
            Union[float, tuple[float]]:
                Metric output or outputs.
        """
        raise NotImplementedError()

    @classmethod
    def normalize(cls, raw_score):
        """Compute the normalized value of the metric.

        Args:
            raw_score (float):
                The value of the metric from `compute`.

        Returns:
            float:
                The normalized value of the metric.
        """
        min_value = float(cls.min_value)
        max_value = float(cls.max_value)

        if max_value < raw_score or min_value > raw_score:
            raise ValueError('`raw_score` must be between `min_value` and `max_value`.')

        is_min_finite = min_value not in (float('-inf'), float('inf'))
        is_max_finite = max_value not in (float('-inf'), float('inf'))

        score = None
        if is_min_finite and is_max_finite:
            score = (raw_score - min_value) / (max_value - min_value)

        elif not is_min_finite and is_max_finite:
            score = np.exp(raw_score - max_value)

        elif is_min_finite and not is_max_finite:
            score = 1.0 - np.exp(min_value - raw_score)

        else:
            score = 1 / (1 + np.exp(-raw_score))

        if score is None or score < 0 or score > 1:
            raise AssertionError(f'This should be unreachable. The score {score} should be'
                                 f'a value between 0 and 1.')

        if cls.goal == Goal.MINIMIZE:
            return 1.0 - score

        return score
