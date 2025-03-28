"""New Row Synthesis metric for single table."""

import warnings

import pandas as pd

from sdmetrics._utils_metadata import _convert_datetime_column
from sdmetrics.errors import IncomputableMetricError
from sdmetrics.goal import Goal
from sdmetrics.single_table.base import SingleTableMetric
from sdmetrics.utils import get_columns_from_metadata, get_type_from_column_meta, strip_characters


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
    def compute_breakdown(
        cls,
        real_data,
        synthetic_data,
        metadata=None,
        numerical_match_tolerance=0.01,
        synthetic_sample_size=None,
    ):
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
                A float larger than 0 representing how close two numerical values have to be
                in order to be considered a match. Defaults to `0.01`.
            synthetic_sample_size (int):
                The number of synthetic rows to sample before computing this metric.
                Use this to speed up the computation time if you have a large amount
                of synthetic data. Note that the final score may not be as precise if
                your sample size is low. Defaults to ``None``, which does not sample,
                and uses all of the provided rows.

        Returns:
            dict:
                The new row synthesis score breakdown.
        """
        real_data, synthetic_data, metadata = cls._validate_inputs(
            real_data, synthetic_data, metadata
        )

        if synthetic_sample_size is not None:
            if synthetic_sample_size > len(synthetic_data):
                warnings.warn(
                    f'The provided `synthetic_sample_size` of {synthetic_sample_size} '
                    'is larger than the number of synthetic data rows '
                    f'({len(synthetic_data)}). Proceeding without sampling.'
                )
            else:
                synthetic_data = synthetic_data.sample(n=synthetic_sample_size)

        for field, field_meta in get_columns_from_metadata(metadata).items():
            if get_type_from_column_meta(field_meta) == 'datetime':
                real_data[field] = _convert_datetime_column(field, real_data[field], field_meta)
                synthetic_data[field] = _convert_datetime_column(
                    field, synthetic_data[field], field_meta
                )

                real_data[field] = pd.to_numeric(real_data[field])
                synthetic_data[field] = pd.to_numeric(synthetic_data[field])

        try:
            numerical_fields = cls._select_fields(metadata, ('numerical', 'datetime'))
        except IncomputableMetricError:
            numerical_fields = []

        try:
            categorical_fields = cls._select_fields(metadata, ('categorical', 'boolean'))
        except IncomputableMetricError:
            categorical_fields = []

        for column_name in real_data.columns:
            new_column_name = strip_characters(['\n', '.', "'"], column_name)
            if new_column_name != column_name:
                if new_column_name in real_data.columns:
                    while new_column_name in real_data.columns:
                        new_column_name += '_'

                real_data = real_data.rename(columns={column_name: new_column_name})
                synthetic_data = synthetic_data.rename(columns={column_name: new_column_name})
                if column_name in numerical_fields:
                    numerical_fields.remove(column_name)
                    numerical_fields.append(new_column_name)
                elif column_name in categorical_fields:
                    categorical_fields.remove(column_name)
                    categorical_fields.append(new_column_name)

        num_unique_rows = 0
        for index, row in synthetic_data.iterrows():
            row_filter = []
            for field in real_data.columns:
                if field not in numerical_fields and field not in categorical_fields:
                    continue

                if pd.isna(row[field]):
                    field_filter = f'`{field}`.isnull()'
                elif field in numerical_fields:
                    field_filter = (
                        f'abs(`{field}` - {row[field]}) <= '
                        f'{abs(numerical_match_tolerance * row[field])}'
                    )
                elif field in categorical_fields:
                    if real_data[field].dtype == 'O':
                        field_filter = f'`{field}` == {repr(row[field])}'
                    else:
                        field_filter = f'`{field}` == {row[field]}'

                row_filter.append(field_filter)

            engine = None
            if len(row_filter) >= 32:  # Limit set by NPY_MAXARGS
                engine = 'python'
            try:
                matches = real_data.query(' and '.join(row_filter), engine=engine)
            except TypeError:
                if len(real_data) > 10000:
                    warnings.warn(
                        'Unable to optimize query. For better formance, set the '
                        '`synthetic_sample_size` parameter or upgrade to Python 3.8'
                    )

                matches = real_data.query(' and '.join(row_filter), engine='python')

            if matches is None or matches.empty:
                num_unique_rows += 1

        return {
            'score': num_unique_rows / len(synthetic_data),
            'num_new_rows': num_unique_rows,
            'num_matched_rows': len(synthetic_data) - num_unique_rows,
        }

    @classmethod
    def compute(
        cls,
        real_data,
        synthetic_data,
        metadata=None,
        numerical_match_tolerance=0.01,
        synthetic_sample_size=None,
    ):
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
                A float larger than 0 representing how close two numerical values have to be
                in order to be considered a match. Defaults to `0.01`.
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
        return cls.compute_breakdown(
            real_data,
            synthetic_data,
            metadata,
            numerical_match_tolerance,
            synthetic_sample_size,
        )['score']

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
