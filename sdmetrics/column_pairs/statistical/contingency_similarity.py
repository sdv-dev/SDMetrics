"""Contingency Similarity Metric."""

import pandas as pd

from sdmetrics.column_pairs.base import ColumnPairsMetric
from sdmetrics.goal import Goal
from sdmetrics.utils import discretize_column


class ContingencySimilarity(ColumnPairsMetric):
    """Contingency similarity metric.

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

    name = 'ContingencySimilarity'
    goal = Goal.MAXIMIZE
    min_value = 0.0
    max_value = 1.0

    @staticmethod
    def _validate_inputs(
        real_data, synthetic_data, continuous_column_names, num_discrete_bins, num_rows_subsample
    ):
        for data in [real_data, synthetic_data]:
            if not isinstance(data, pd.DataFrame) or len(data.columns) != 2:
                raise ValueError('The data must be a pandas DataFrame with two columns.')

        if set(real_data.columns) != set(synthetic_data.columns):
            raise ValueError('The columns in the real and synthetic data must match.')

        if continuous_column_names is not None:
            bad_continuous_columns = "' ,'".join([
                column for column in continuous_column_names if column not in real_data.columns
            ])
            if bad_continuous_columns:
                raise ValueError(
                    f"Continuous column(s) '{bad_continuous_columns}' not found in the data."
                )

        if not isinstance(num_discrete_bins, int) or num_discrete_bins <= 0:
            raise ValueError('`num_discrete_bins` must be an integer greater than zero.')

        if num_rows_subsample is not None:
            if not isinstance(num_rows_subsample, int) or num_rows_subsample <= 0:
                raise ValueError('`num_rows_subsample` must be an integer greater than zero.')

    @classmethod
    def compute_breakdown(
        cls,
        real_data,
        synthetic_data,
        continuous_column_names=None,
        num_discrete_bins=10,
        num_rows_subsample=None,
    ):
        """Compute the breakdown of this metric."""
        cls._validate_inputs(
            real_data,
            synthetic_data,
            continuous_column_names,
            num_discrete_bins,
            num_rows_subsample,
        )
        columns = real_data.columns[:2]

        if num_rows_subsample is not None:
            real_data = real_data.sample(min(num_rows_subsample, len(real_data)))
            synthetic_data = synthetic_data.sample(min(num_rows_subsample, len(synthetic_data)))

        real = real_data[columns]
        synthetic = synthetic_data[columns]
        if continuous_column_names:
            for column in continuous_column_names:
                real[column], synthetic[column] = discretize_column(
                    real[column], synthetic[column], num_discrete_bins=num_discrete_bins
                )

        contingency_real = real.groupby(list(columns), dropna=False).size() / len(real)
        contingency_synthetic = synthetic.groupby(list(columns), dropna=False).size() / len(
            synthetic
        )
        combined_index = contingency_real.index.union(contingency_synthetic.index, sort=False)
        contingency_synthetic = contingency_synthetic.reindex(combined_index, fill_value=0)
        contingency_real = contingency_real.reindex(combined_index, fill_value=0)
        diff = abs(contingency_real - contingency_synthetic).fillna(0)
        variation = diff / 2
        return {'score': 1 - variation.sum()}

    @classmethod
    def compute(
        cls,
        real_data,
        synthetic_data,
        continuous_column_names=None,
        num_discrete_bins=10,
        num_rows_subsample=None,
    ):
        """Compare the contingency similarity of two discrete columns.

        Args:
            real_data (pd.DataFrame):
                The values from the real dataset.
            synthetic_data (pd.DataFrame):
                The values from the synthetic dataset.
            continuous_column_names (list[str], optional):
                The list of columns to discretize before running the metric. The column names in
                this list should match the column names in the real and synthetic data. Defaults
                to ``None``.
            num_discrete_bins (int, optional):
                The number of bins to create for the continuous columns. Defaults to 10.
            num_rows_subsample (int, optional):
                The number of rows to subsample from the real and synthetic data before computing
                the metric. Defaults to ``None``.

        Returns:
            float:
                The contingency similarity of the two columns.
        """
        return cls.compute_breakdown(
            real_data,
            synthetic_data,
            continuous_column_names,
            num_discrete_bins,
            num_rows_subsample,
        )['score']

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
