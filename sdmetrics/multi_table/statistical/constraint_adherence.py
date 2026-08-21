"""Constraint Adherence metric."""
import warnings

import numpy as np
import pandas as pd

from sdmetrics.goal import Goal
from sdmetrics.multi_table.base import MultiTableMetric
from sdmetrics.multi_table.statistical.constraints.base import BaseConstraint
from sdmetrics.multi_table.statistical.constraints.error import ConstraintNotApplicableError


class ConstraintAdherence(MultiTableMetric):
    """Constraint Adherence metric.

    Compute the fraction of data points in the synthetic data that
    follow the specified constraint.

    Attributes:
        name (str):
            Name to use when reports about this metric are printed.
        goal (sdmetrics.goal.Goal):
            The goal of this metric.
        min_value (float):
            Minimum value that this metric can take.
        max_value (float):
            Maximum value that this metric can take.
    """

    name = 'ConstraintAdherence'
    goal = Goal.MAXIMIZE
    min_value = 0.0
    max_value = 1.0

    @classmethod
    def compute(cls, real_data, synthetic_data, metadata, constraint):
        """Compute the percentage of rows that adhere to the constraint.

        Args:
            real_data (dict[str, pandas.DataFrame]):
                The tables from the real dataset, passed as a dictionary of
                table names and pandas.DataFrames.
            synthetic_data (dict[str, pandas.DataFrame]):
                The tables from the synthetic dataset, passed as a dictionary of
                table names and pandas.DataFrames.
            metadata (dict):
                Multi-table metadata dict.
            constraint (dict):
                A dictionary that defines the type of constraint and its parameters.

        Returns:
            float:
                The proportion of data points in the synthetic data that match
                the specified constraint format.
        """
        try:
            constraint = BaseConstraint.load_constraint_from_dict(constraint)
        except ValueError as error:
            warnings.warn(f'Unable to check the constraint: {error}')
            return np.nan

        try:
            real_score = constraint.get_score(real_data, metadata)
        except ConstraintNotApplicableError as error:
            warnings.warn(f'Unable to check the constraint against the real data: {error}')
            real_score = np.nan

        if not pd.isna(real_score) and real_score < 1.0:
            warnings.warn(
                f"The real data does not adhere to the '{constraint.__class__.__name__}' "
                f'constraint ({round(real_score * 100, 2)}% of the rows are valid). '
            )

        try:
            return constraint.get_score(synthetic_data, metadata)
        except ConstraintNotApplicableError as error:
            warnings.warn(
                f'Unable to check the constraint against the synthetic data: {error}'
            )
            return np.nan
