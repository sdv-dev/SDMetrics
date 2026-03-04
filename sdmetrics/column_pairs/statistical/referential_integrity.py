"""Referential Integrity Metric."""

import logging

import pandas as pd

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
    INDICATOR_NAME = '__ri_indicator__'

    @staticmethod
    def _create_unique_name(name, list_names):
        """Modify the ``name`` parameter if it already exists in the list of names."""
        result = name
        while result in list_names:
            result += '_'

        return result

    @classmethod
    def compute_breakdown(cls, real_data, synthetic_data):
        """Compute the score breakdown of the referential integrity metric.

        Args:
            real_data (tuple of 2 pandas.DataFrame):
                (primary_key, foreign_key) columns from the real data.
            synthetic_data (tuple of 2 pandas.DataFrame):
                (primary_key, foreign_key) columns from the synthetic data.

        Returns:
            dict:
                The score breakdown of the key uniqueness metric.
        """
        real_pk_df, real_fk_df = real_data
        synth_pk_df, synth_fk_df = synthetic_data
        pk_columns = list(real_pk_df.columns)
        fk_columns = list(real_fk_df.columns)
        indicator_name = cls._create_unique_name(cls.INDICATOR_NAME, pk_columns + fk_columns)

        real_merged = real_fk_df.merge(
            real_pk_df.drop_duplicates(),
            how='left',
            left_on=fk_columns,
            right_on=pk_columns,
            indicator=indicator_name,
        )
        missing_parents = (real_merged[indicator_name] == 'left_only').any()
        if missing_parents:
            LOGGER.info("The real data has foreign keys that don't reference any primary key.")

        if len(fk_columns) == 1 and pd.isna(real_fk_df[fk_columns[0]]).any():
            synth_fk_df = synth_fk_df.dropna()

        synth_merged = synth_fk_df.merge(
            synth_pk_df.drop_duplicates(),
            how='left',
            left_on=fk_columns,
            right_on=pk_columns,
            indicator=indicator_name,
        )

        score = (synth_merged[indicator_name] == 'both').mean()
        return {'score': score}

    @classmethod
    def compute(cls, real_data, synthetic_data):
        """Compute the referential integrity of two columns.

        Args:
            real_data (tuple of 2 pandas.DataFrame):
                (primary_key, foreign_key) columns from the real data.
            synthetic_data (tuple of 2 pandas.DataFrame):
                (primary_key, foreign_key) columns from the synthetic data.

        Returns:
            float:
                The key uniqueness of the two columns.
        """
        return cls.compute_breakdown(real_data, synthetic_data)['score']
