"""New Row Synthesis metric for single table."""
import warnings

import numpy as np
import pandas as pd

from sdmetrics.goal import Goal
from sdmetrics.single_table.base import SingleTableMetric


class NewRowSynthesis(SingleTableMetric):
    """NewRowSynthesis Single Table metric.

    This metric measures whether each row in the synthetic data is new,
    or whether it exactly matches a row in the real data.

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

    name = 'NewRowSynthesis'
    goal = Goal.MAXIMIZE
    min_value = 0
    max_value = 1

    @classmethod
    def compute(cls, real_data, synthetic_data, metadata=None, numerical_match_tolerance=0.01,
                synthetic_sample_size=None):
        """Compute this metric.

        This metric looks for matches between the real and synthetic data for
        the compatible columns. This metric also looks for matches in missing values.

        Args:
            real_data (Union[numpy.ndarray, pandas.DataFrame]):
                The values from the real dataset.
            synthetic_data (Union[numpy.ndarray, pandas.DataFrame]):
                The values from the synthetic dataset.
            metadata (dict):
                Table metadata dict.
            numerical_match_tolerance (float):
                A float >0.0 representing how close two numerical values have to be
                in order to be considered a match.
            synthetic_sample_size (int):
                The number of synthetic rows to sample before computing this metric.
                Use this to speed up the computation time if you have a large amount
                of synthetic data. Note that the final score may not be as precise if
                your sample size is low. Defaults to ``None``, which does not sample,
                and uses all of the provided rows.

        Returns:
            float:
                The new row synthesis score.
        """
        real_data, synthetic_data, metadata = cls._validate_inputs(
            real_data, synthetic_data, metadata)

        if synthetic_sample_size is not None:
            if synthetic_sample_size > len(synthetic_data):
                warnings.warn(f'The provided `synthetic_sample_size` of {synthetic_sample_size} '
                              'is larger than the number of synthetic data rows '
                              f'({len(synthetic_data)}). Proceeding without sampling.')
            else:
                synthetic_data = synthetic_data.sample(n=synthetic_sample_size)

        numerical_fields = []
        discrete_fields = []
        for field, field_meta in metadata['fields'].items():
            if field_meta['type'] == 'datetime':
                real_data[field] = pd.to_datetime(real_data[field])
                synthetic_data[field] = pd.to_datetime(synthetic_data[field])
                numerical_fields.append(field)
            elif field_meta['type'] == 'numerical':
                numerical_fields.append(field)
            else:
                discrete_fields.append(field)

        num_unique_rows = 0
        for index, row in synthetic_data.iterrows():
            row_filter = []
            for field in real_data.columns:
                if field in numerical_fields:
                    field_filter = f'{field}.isnull()' if np.isnan(row[field]) else (
                        f'abs({field} - {row[field]}) < {numerical_match_tolerance * row[field]}')
                    row_filter.append(field_filter)
                else:
                    row_filter.append(f"{field} == '{row[field]}'")

            matches = real_data.query(' and '.join(row_filter))
            if matches is None or matches.empty:
                num_unique_rows += 1

        return num_unique_rows / len(synthetic_data)

    @classmethod
    def normalize(cls, raw_score):
        """Normalize the log-likelihood value.

        Notice that this is not the mean likelihood.

        Args:
            raw_score (float):
                The value of the metric from `compute`.

        Returns:
            float:
                The normalized value of the metric
        """
        return super().normalize(raw_score)
