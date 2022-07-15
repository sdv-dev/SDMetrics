"""The CardinalityStatisticSimilarity metric."""

import numpy as np
import pandas as pd

from sdmetrics.goal import Goal
from sdmetrics.multi_table.base import MultiTableMetric


class CardinalityStatisticSimilarity(MultiTableMetric):
    """CardinalityStatisticSimilarity metric computes the similarity of the cardinality.

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

    name = 'CardinalityStatisticSimilarity'
    goal = Goal.MAXIMIZE
    min_value = 0.0
    max_value = 1.0

    @classmethod
    def _get_cardinality_distribution(cls, parent_column, child_column):
        """Compute the cardinality distribution of the (parent, child) pairing.

        Args:
            parent_column (pandas.Series):
                The parent column.
            child_column (pandas.Series):
                The child column.

        Returns:
            pandas.Series:
                The cardinality distribution.
        """
        child_df = pd.DataFrame({'child_counts': child_column.value_counts()})
        cardinality_df = pd.DataFrame({'parent': parent_column}).join(
            child_df, on='parent').fillna(0)

        return cardinality_df['child_counts']

    @classmethod
    def _compute_statistic(cls, real_distribution, synthetic_distribution, statistic):
        """Compute the requested statistic over the two distributions.

        Args:
            real_distribution (pandas.Series):
                The distribution of the real data.
            synthetic_distribution (pandas.Series):
                The distribution of the synthetic data.
            statistic (str):
                The desired statistic to compute. Must be either 'mean', 'median', or 'std'.

        Returns:
            dict:
                A score breakdown of the real, synthetic, and comparison scores.
        """
        if statistic == 'mean':
            score_real = real_distribution.mean()
            score_synthetic = synthetic_distribution.mean()
        elif statistic == 'median':
            score_real = real_distribution.median()
            score_synthetic = synthetic_distribution.median()
        elif statistic == 'std':
            score_real = real_distribution.std()
            score_synthetic = synthetic_distribution.std()
        else:
            raise ValueError(f'requested statistic {statistic} is not valid. '
                             'Please choose either mean, std, or median.')

        score = 1 - abs(score_real - score_synthetic) / (
            real_distribution.max() - real_distribution.min())

        return {'real': score_real, 'synthetic': score_synthetic, 'score': max(score, 0)}

    @classmethod
    def compute_breakdown(cls, real_data, synthetic_data, metadata=None, statistic='mean'):
        """Compute the breakdown of cardinality statistic similarity in the given tables.

        Compute the cardinality distributions for the real and synthetic data for each
        (parent, child) relationship specified in the metadata. Then compute the requested
        statistic over the two cardinality distributions, and compare the final statistic values
        to obtain the cardinality statistic similarity score.

        Args:
            real_data (dict[str, pandas.DataFrame]):
                The tables from the real dataset, passed as a dictionary of
                table names and pandas.DataFrames.
            synthetic_data (dict[str, pandas.DataFrame]):
                The tables from the synthetic dataset, passed as a dictionary of
                table names and pandas.DataFrames.
            metadata (dict):
                Multi-table metadata dict. If not passed, it is build based on the
                real_data fields and dtypes.
            statistic (str):
                The desired statistic to compute. Must be either 'mean', 'median', or 'std'.

        Returns:
            dict:
                A dict mapping (parent, child) values to the score breakdown for the
                cardinality distributions of that foreign key.
        """
        if set(real_data.keys()) != set(synthetic_data.keys()):
            raise ValueError('`real_data` and `synthetic_data` must have the same tables.')
        if metadata is None:
            raise ValueError('`metadata` cannot be ``None``.')
        if not isinstance(metadata, dict):
            metadata = metadata.to_dict()

        score_breakdowns = {}
        for table_name, table in metadata['tables']:
            for field_name, field in table['fields']:
                if 'ref' in field:
                    child_table_name = field['ref']['table']
                    child_field_name = field['ref']['field']
                    cardinality_real = cls._get_cardinality_distribution(
                        real_data[table_name][field_name],
                        real_data[child_table_name][child_field_name],
                    )
                    cardinality_synthetic = cls._get_cardinality_distribution(
                        synthetic_data[table_name][field_name],
                        synthetic_data[child_table_name][child_field_name],
                    )
                    score_breakdown = cls._compute_statistic(
                        cardinality_real, cardinality_synthetic, statistic)
                    score_breakdowns[(table_name, child_table_name)] = score_breakdown

        if len(score_breakdowns) == 0:
            return {'score': np.nan}

        return score_breakdowns

    @classmethod
    def compute(cls, real_data, synthetic_data, metadata=None, statistic='mean'):
        """Compute the average of cardinality statistic similarity in the given tables.

        Compute the average statistic similarity in cardinality distributions for
        (parent, child) relationship in the given tables, as specified in the metadata.
        The statistic similarity is computed based on the requested statistic.

        Args:
            real_data (dict[str, pandas.DataFrame]):
                The tables from the real dataset, passed as a dictionary of
                table names and pandas.DataFrames.
            synthetic_data (dict[str, pandas.DataFrame]):
                The tables from the synthetic dataset, passed as a dictionary of
                table names and pandas.DataFrames.
            metadata (dict):
                Multi-table metadata dict. If not passed, it is build based on the
                real_data fields and dtypes.
            statistic (str):
                The desired statistic to compute. Must be either 'mean', 'median', or 'std'.

        Returns:
            float:
                The average of all (parent, child) cardinality statistic similarity scores.
        """
        score_breakdowns = cls.compute_breakdown(real_data, synthetic_data, metadata, statistic)
        all_scores = [breakdown['score'] for breakdown in score_breakdowns]

        return sum(all_scores) / len(all_scores)

    @classmethod
    def normalize(cls, raw_score):
        """Return the `raw_score` as is, since it is already normalized.

        Args:
            raw_score (float):
                The value of the metric from ``compute``.

        Returns:
            float:
                The normalized value of the metric.
        """
        return super().normalize(raw_score)
