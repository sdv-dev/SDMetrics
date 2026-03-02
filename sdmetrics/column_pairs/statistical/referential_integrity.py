"""Referential Integrity Metric."""

import logging

from sdmetrics.column_pairs.base import ColumnPairsMetric
from sdmetrics.goal import Goal

LOGGER = logging.getLogger(__name__)
INDICATOR_NAME = '__ri_indicator__'


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

        # Check if real data has broken referential integrity
        real_merged = real_fk_df.merge(
            real_pk_df.drop_duplicates(),
            how='left',
            left_on=fk_columns,
            right_on=pk_columns,
            indicator=INDICATOR_NAME,
        )
        missing_parents = (real_merged[INDICATOR_NAME] == 'left_only').any()
        if missing_parents:
            LOGGER.info("The real data has foreign keys that don't reference any primary key.")

        if real_fk_df.isna().any().any():
            synth_fk_df = synth_fk_df.dropna()

        synth_merged = synth_fk_df.merge(
            synth_pk_df.drop_duplicates(),
            how='left',
            left_on=fk_columns,
            right_on=pk_columns,
            indicator=INDICATOR_NAME,
        )

        score = (synth_merged[INDICATOR_NAME] == 'both').mean()
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
