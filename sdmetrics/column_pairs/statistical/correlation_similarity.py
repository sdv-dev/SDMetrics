"""Correlation Similarity Metric."""

import warnings

import numpy as np
import pandas as pd
from scipy.stats import pearsonr, spearmanr

from sdmetrics.column_pairs.base import ColumnPairsMetric
from sdmetrics.goal import Goal
from sdmetrics.utils import is_datetime
from sdmetrics.warnings import ConstantInputWarning


class CorrelationSimilarity(ColumnPairsMetric):
    """Correlation similarity metric.

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

    name = 'CorrelationSimilarity'
    goal = Goal.MAXIMIZE
    min_value = 0.0
    max_value = 1.0

    @staticmethod
    def _generate_warning_msg(columns, prefix, warning_messages):
        if len(columns) > 1:
            cols = ', '.join(columns)
            warning_messages.append(
                f"The {prefix} in columns '{cols}' contain a constant value. "
                'Correlation is undefined for constant data.'
            )

        elif len(columns):
            warning_messages.append(
                f"The {prefix} in column '{columns[0]}' contains a constant value. "
                'Correlation is undefined for constant data.'
            )

    @classmethod
    def compute_breakdown(cls, real_data, synthetic_data, coefficient='Pearson'):
        """Compare the breakdown of correlation similarity of two continuous columns.

        Args:
            real_data (Union[numpy.ndarray, pandas.Series]):
                The values from the real dataset.
            synthetic_data (Union[numpy.ndarray, pandas.Series]):
                The values from the synthetic dataset.

        Returns:
            dict:
                A dict containing the score, and the real and synthetic metric values.
        """
        real_data = real_data.copy()
        synthetic_data = synthetic_data.copy()

        if not isinstance(real_data, pd.DataFrame):
            real_data = pd.DataFrame(real_data)
            synthetic_data = pd.DataFrame(synthetic_data)

        if (real_data.nunique() == 1).any() or (synthetic_data.nunique() == 1).any():
            warning_messages = []
            real_columns = list(real_data.loc[:, real_data.nunique() == 1].columns)
            synthetic_columns = list(synthetic_data.loc[:, synthetic_data.nunique() == 1].columns)
            cls._generate_warning_msg(real_columns, 'real data', warning_messages)
            cls._generate_warning_msg(synthetic_columns, 'synthetic data', warning_messages)

            for msg in warning_messages:
                warnings.warn(ConstantInputWarning(msg))

            return {'score': np.nan}

        real_data = real_data.dropna()
        synthetic_data = synthetic_data.dropna()
        column1, column2 = real_data.columns[:2]

        if is_datetime(real_data[column1]):
            real_data[column1] = pd.to_numeric(real_data[column1])
            synthetic_data[column1] = pd.to_numeric(synthetic_data[column1])

        if is_datetime(real_data[column2]):
            real_data[column2] = pd.to_numeric(real_data[column2])
            synthetic_data[column2] = pd.to_numeric(synthetic_data[column2])

        correlation_fn = None
        if coefficient == 'Pearson':
            correlation_fn = pearsonr
        elif coefficient == 'Spearman':
            correlation_fn = spearmanr
        else:
            raise ValueError(f'requested coefficient {coefficient} is not valid. '
                             'Please choose either Pearson or Spearman.')

        correlation_real, _ = correlation_fn(real_data[column1], real_data[column2])
        correlation_synthetic, _ = correlation_fn(synthetic_data[column1], synthetic_data[column2])

        if np.isnan(correlation_real) or np.isnan(correlation_synthetic):
            return {'score': np.nan}

        return {
            'score': 1 - abs(correlation_real - correlation_synthetic) / 2,
            'real': correlation_real,
            'synthetic': correlation_synthetic,
        }

    @classmethod
    def compute(cls, real_data, synthetic_data, coefficient='Pearson'):
        """Compare the correlation similarity of two continuous columns.

        Args:
            real_data (Union[numpy.ndarray, pandas.Series]):
                The values from the real dataset.
            synthetic_data (Union[numpy.ndarray, pandas.Series]):
                The values from the synthetic dataset.

        Returns:
            float:
                The correlation similarity of the two columns.
        """
        return cls.compute_breakdown(real_data, synthetic_data, coefficient)['score']

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
