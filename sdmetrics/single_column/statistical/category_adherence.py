"""Category Adherence Metric."""

import numpy as np
import pandas as pd

from sdmetrics.goal import Goal
from sdmetrics.single_column.base import SingleColumnMetric


class CategoryAdherence(SingleColumnMetric):
    """Category adherence metric.

    The proportion of synthetic data points that match an existing category from the real data.
    If any of ``range_values`` or ``range_is_nullable`` are provided, they are used instead of
    the categories computed from the real data.

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

    name = 'CategoryAdherence'
    goal = Goal.MAXIMIZE
    min_value = 0.0
    max_value = 1.0

    @classmethod
    def compute_breakdown(
        cls, real_data, synthetic_data, range_values=None, range_is_nullable=None
    ):
        """Compute the score breakdown of the category adherence metric.

        Args:
            real_data (pandas.Series):
                The real data.
            synthetic_data (pandas.Series):
                The synthetic data.
            range_values (List[str], optional):
                The list of categories the column is allowed to take. If ``None``, the
                categories of the real data are used instead. Defaults to ``None``.
            range_is_nullable (bool, optional):
                Whether the column is allowed to contain missing values.
                Defaults to ``None``.

        Returns:
            dict:
                The score breakdown of the category adherence metric.
        """
        real_data = pd.Series(real_data).fillna(np.nan)
        synthetic_data = pd.Series(synthetic_data).fillna(np.nan)
        if range_values is None:
            valid_values = pd.Series(real_data.unique())
        else:
            valid_values = pd.Series(list(range_values))

        if range_is_nullable is None:
            range_is_nullable = any(pd.isna(real_data)) or any(pd.isna(valid_values))

        valid = synthetic_data.isin(valid_values.dropna())
        if range_is_nullable:
            valid = valid | pd.isna(synthetic_data)

        return {'score': valid.mean()}

    @classmethod
    def compute(cls, real_data, synthetic_data, range_values=None, range_is_nullable=None):
        """Compute the category adherence of two columns.

        Args:
            real_data (pandas.Series):
                The real data.
            synthetic_data (pandas.Series):
                The synthetic data.
            range_values (List[str], optional):
                The list of categories the column is allowed to take. If ``None``, the
                categories of the real data are used instead. Defaults to ``None``.
            range_is_nullable (bool, optional):
                Whether the column is allowed to contain missing values.
                Defaults to ``None``.

        Returns:
            float:
                The category adherence metric score.
        """
        return cls.compute_breakdown(
            real_data,
            synthetic_data,
            range_values=range_values,
            range_is_nullable=range_is_nullable,
        )['score']
