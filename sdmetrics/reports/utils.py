"""Report utility methods."""

import copy
import itertools
import warnings

import numpy as np
import pandas as pd

from sdmetrics._utils_metadata import _convert_datetime_column
from sdmetrics.utils import (
    discretize_column,
    get_alternate_keys,
    get_columns_from_metadata,
    get_type_from_column_meta,
)

CONTINUOUS_SDTYPES = ['numerical', 'datetime']
DISCRETE_SDTYPES = ['categorical', 'boolean']


class PlotConfig:
    """Custom plot settings for visualizations."""

    GREEN = '#36B37E'
    RED = '#FF0000'
    ORANGE = '#F16141'
    DATACEBO_DARK = '#000036'
    DATACEBO_GREEN = '#01E0C9'
    DATACEBO_BLUE = '#03AFF1'
    BACKGROUND_COLOR = '#F5F5F8'
    DATACEBO_DARK_TRANSPARENT = 'rgba(0, 0, 54, 0.25)'
    DATACEBO_GREEN_TRANSPARENT = 'rgba(1, 224, 201, 0.25)'
    FONT_SIZE = 18


def discretize_table_data(real_data, synthetic_data, metadata):
    """Create a copy of the real and synthetic data with discretized data.

    Convert numerical and datetime columns to discrete values, and label them
    as categorical.

    Args:
        real_data (pandas.DataFrame):
            The real data.
        synthetic_data (pandas.DataFrame):
            The synthetic data.
        metadata (dict)
            The metadata.

    Returns:
        (pandas.DataFrame, pandas.DataFrame, dict):
            The binned real and synthetic data, and the updated metadata.
    """
    binned_real = real_data.copy()
    binned_synthetic = synthetic_data.copy()
    binned_metadata = copy.deepcopy(metadata)

    for column_name, column_meta in get_columns_from_metadata(metadata).items():
        sdtype = get_type_from_column_meta(column_meta)

        if sdtype in ('numerical', 'datetime'):
            real_col = real_data[column_name]
            synthetic_col = synthetic_data[column_name]
            if sdtype == 'datetime':
                real_col = _convert_datetime_column(column_name, real_col, column_meta)
                synthetic_col = _convert_datetime_column(column_name, synthetic_col, column_meta)

                real_col = pd.to_numeric(real_col)
                synthetic_col = pd.to_numeric(synthetic_col)

            binned_real_col, binned_synthetic_col = discretize_column(real_col, synthetic_col)

            binned_real[column_name] = binned_real_col
            binned_synthetic[column_name] = binned_synthetic_col
            get_columns_from_metadata(binned_metadata)[column_name] = {'sdtype': 'categorical'}

    return binned_real, binned_synthetic, binned_metadata


def _get_non_id_columns(metadata, binned_metadata):
    valid_sdtypes = ['numerical', 'categorical', 'boolean', 'datetime']
    alternate_keys = get_alternate_keys(metadata)
    non_id_columns = []
    for column, column_meta in get_columns_from_metadata(binned_metadata).items():
        is_key = column == metadata.get('primary_key', '') or column in alternate_keys
        if get_type_from_column_meta(column_meta) in valid_sdtypes and not is_key:
            non_id_columns.append(column)

    return non_id_columns


def discretize_and_apply_metric(real_data, synthetic_data, metadata, metric, keys_to_skip=[]):
    """Discretize the data and apply the given metric.

    Args:
        real_data (pandas.DataFrame):
            The real data.
        synthetic_data (pandas.DataFrame):
            The synthetic data.
        metadata (dict)
            The metadata.
        metric (sdmetrics.single_table.MultiColumnPairMetric):
            The column pair metric to apply.
        keys_to_skip (list[tuple(str)] or None):
            A list of keys for which to skip computing the metric.

    Returns:
        dict:
            The metric results.
    """
    metric_results = {}

    binned_real, binned_synthetic, binned_metadata = discretize_table_data(
        real_data, synthetic_data, metadata
    )

    non_id_cols = _get_non_id_columns(metadata, binned_metadata)
    for columns in itertools.combinations(non_id_cols, r=2):
        sorted_columns = tuple(sorted(columns))
        if (
            sorted_columns not in keys_to_skip
            and (sorted_columns[1], sorted_columns[0]) not in keys_to_skip
        ):
            result = metric.column_pairs_metric.compute_breakdown(
                binned_real[list(sorted_columns)],
                binned_synthetic[list(sorted_columns)],
            )
            metric_results[sorted_columns] = result
            metric_results[sorted_columns] = result

    return metric_results


def aggregate_metric_results(metric_results):
    """Aggregate the scores and errors in a metric results mapping.

    Args:
        metric_results (dict):
            The metric results to aggregate.

    Returns:
        (float, int):
            The average of the metric scores, and the number of errors.
    """
    if len(metric_results) == 0:
        return np.nan, 0

    metric_scores = []
    num_errors = 0

    for _, breakdown in metric_results.items():
        metric_score = breakdown.get('score', np.nan)
        if not np.isnan(metric_score):
            metric_scores.append(metric_score)
        if 'error' in breakdown:
            num_errors += 1

    return np.mean(metric_scores), num_errors


def _validate_categorical_values(real_data, synthetic_data, metadata, table=None):
    """Get categorical values found in synthetic data but not real data for all columns.

    Args:
        real_data (pd.DataFrame):
            The real data.
        synthetic_data (pd.DataFrame):
            The synthetic data.
        metadata (dict):
            The metadata.
        table (str, optional):
            The name of the current table, if one exists
    """
    if table:
        warning_format = (
            f'Unexpected values ({{values}}) in column "{{column}}" and table "{table}"'
        )
    else:
        warning_format = 'Unexpected values ({values}) in column "{column}"'

    columns = get_columns_from_metadata(metadata)
    for column, column_meta in columns.items():
        column_type = get_type_from_column_meta(column_meta)
        if column_type == 'categorical':
            extra_categories = [
                value
                for value in synthetic_data[column].unique()
                if value not in real_data[column].unique()
            ]
            if extra_categories:
                value_list = '", "'.join(str(value) for value in extra_categories[:5])
                values = (
                    f'"{value_list}" + more' if len(extra_categories) > 5 else f'"{value_list}"'
                )
                warnings.warn(warning_format.format(values=values, column=column))
