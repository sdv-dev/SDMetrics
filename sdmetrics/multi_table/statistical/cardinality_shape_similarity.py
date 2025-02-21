"""The CardinalityShapeSimilarity metric."""

import numpy as np
from scipy.stats import ks_2samp

from sdmetrics._utils_metadata import (
    _validate_metadata_dict,
)
from sdmetrics.goal import Goal
from sdmetrics.multi_table.base import MultiTableMetric
from sdmetrics.utils import get_cardinality_distribution


class CardinalityShapeSimilarity(MultiTableMetric):
    """CardinalityShapeSimilarity metric computes the similarity of the cardinality.

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

    name = 'CardinalityShapeSimilarity'
    goal = Goal.MAXIMIZE
    min_value = 0.0
    max_value = 1.0

    @classmethod
    def compute_breakdown(cls, real_data, synthetic_data, metadata):
        """Compute the breakdown of cardinality shape similarity in the given tables.

        Compute the cardinality distributions for the real and synthetic data for each
        (parent, child) relationship specified in the metadata. Then compute the `KSComplement`
        over the two cardinality distributions, and compare the final scores
        to obtain the cardinality shape similarity score.

        Args:
            real_data (dict[str, pandas.DataFrame]):
                The tables from the real dataset, passed as a dictionary of
                table names and pandas.DataFrames.
            synthetic_data (dict[str, pandas.DataFrame]):
                The tables from the synthetic dataset, passed as a dictionary of
                table names and pandas.DataFrames.
            metadata (dict):
                Multi-table metadata dict.

        Returns:
            dict:
                A dict mapping (parent, child) values to the score breakdown for the
                cardinality distributions of that foreign key.
        """
        if set(real_data.keys()) != set(synthetic_data.keys()):
            raise ValueError('`real_data` and `synthetic_data` must have the same tables.')

        _validate_metadata_dict(metadata)
        score_breakdowns = {}
        for rel in metadata.get('relationships', []):
            cardinality_real = get_cardinality_distribution(
                real_data[rel['parent_table_name']][rel['parent_primary_key']],
                real_data[rel['child_table_name']][rel['child_foreign_key']],
            )
            cardinality_synthetic = get_cardinality_distribution(
                synthetic_data[rel['parent_table_name']][rel['parent_primary_key']],
                synthetic_data[rel['child_table_name']][rel['child_foreign_key']],
            )
            statistic, _ = ks_2samp(cardinality_real, cardinality_synthetic)
            score_breakdowns[(rel['parent_table_name'], rel['child_table_name'])] = {
                'score': 1 - statistic
            }

        if len(score_breakdowns) == 0:
            return {'score': np.nan}

        return score_breakdowns

    @classmethod
    def compute(cls, real_data, synthetic_data, metadata):
        """Compute the average of cardinality shape similarity in the given tables.

        Compute the average shape similarity in cardinality distributions for
        (parent, child) relationship in the given tables, as specified in the metadata.
        The shape similarity is computed based on the `KSComplement`.

        Args:
            real_data (dict[str, pandas.DataFrame]):
                The tables from the real dataset, passed as a dictionary of
                table names and pandas.DataFrames.
            synthetic_data (dict[str, pandas.DataFrame]):
                The tables from the synthetic dataset, passed as a dictionary of
                table names and pandas.DataFrames.
            metadata (dict):
                Multi-table metadata dict.

        Returns:
            float:
                The average of all (parent, child) cardinality statistic similarity scores.
        """
        score_breakdowns = cls.compute_breakdown(real_data, synthetic_data, metadata)
        if 'score' in score_breakdowns:
            return score_breakdowns['score']

        all_scores = [breakdown['score'] for _, breakdown in score_breakdowns.items()]

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
