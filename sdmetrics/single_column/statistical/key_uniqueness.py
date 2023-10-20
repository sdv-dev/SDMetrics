"""Key Uniqueness Metric."""

from sdmetrics.errors import InvalidDataError
from sdmetrics.goal import Goal
from sdmetrics.single_column.base import SingleColumnMetric


class KeyUniqueness(SingleColumnMetric):
    """Key uniqueness metric.

    Compute the fraction of rows in the synthetic data that are unique.

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

    name = 'KeyUniqueness'
    goal = Goal.MAXIMIZE
    min_value = 0.0
    max_value = 1.0

    @classmethod
    def compute_breakdown(cls, real_data, synthetic_data):
        """Compute the score breakdown of the key uniqueness metric.

        Args:
            real_data (pandas.Series):
                The real data.
            synthetic_data (pandas.Series):
                The synthetic data.

        Returns:
            dict:
                The score breakdown of the key uniqueness metric.
        """
        has_duplicates = real_data.duplicated().any()
        if has_duplicates:
            raise InvalidDataError('The real data contains NA or duplicate values.')

        duplicates_synthetic = synthetic_data.duplicated()
        score = 1 - duplicates_synthetic.sum() / len(synthetic_data)

        return {'score': score}

    @classmethod
    def compute(cls, real_data, synthetic_data):
        """Compute the key uniqueness of two columns.

        Args:
            real_data (pandas.Series):
                The real data.
            synthetic_data (pandas.Series):
                The synthetic data.

        Returns:
            float:
                The key uniqueness of the two columns.
        """
        return cls.compute_breakdown(real_data, synthetic_data)['score']

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
        return super().normalize(raw_score)
