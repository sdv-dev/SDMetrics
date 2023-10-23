"""Referential Integrity Metric."""
import logging

from sdmetrics.column_pairs.base import ColumnPairsMetric
from sdmetrics.goal import Goal

LOGGER = logging.getLogger(__name__)


class ReferentialIntegrity(ColumnPairsMetric):
    """Referential Integrity metric.

    Compute the fraction of foreign key values that reference a value in the primary key column
    in the synthetic data.

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

    name = 'ReferentialIntegrity'
    goal = Goal.MAXIMIZE
    min_value = 0.0
    max_value = 1.0

    @classmethod
    def compute_breakdown(cls, real_data, synthetic_data):
        """Compute the score breakdown of the referential integrity metric.

        Args:
            real_data (tuple of 2 pandas.Series):
                (primary_key, foreign_key) columns from the real data.
            synthetic_data (tuple of 2 pandas.Series):
                (primary_key, foreign_key) columns from the synthetic data.

        Returns:
            dict:
                The score breakdown of the key uniqueness metric.
        """
        missing_parents = not real_data[1].isin(real_data[0]).all()
        if missing_parents:
            LOGGER.info(
                "The real data has foreign keys that don't reference any primary key."
            )

        score = synthetic_data[1].isin(synthetic_data[0]).mean()

        return {'score': score}

    @classmethod
    def compute(cls, real_data, synthetic_data):
        """Compute the referential integrity of two columns.

        Args:
            real_data (tuple of 2 pandas.Series):
                (primary_key, foreign_key) columns from the real data.
            synthetic_data (tuple of 2 pandas.Series):
                (primary_key, foreign_key) columns from the synthetic data.

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
