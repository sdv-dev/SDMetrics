"""Distance to closest record measurement functions."""

import numpy as np
import pandas as pd

from sdmetrics._utils_metadata import _process_data_with_metadata
from sdmetrics.utils import get_columns_from_metadata

CHUNK_SIZE = 1000


def _process_dcr_chunk(dataset_chunk, reference_chunk, cols_to_keep, metadata, ranges):
    full_dataset = dataset_chunk.merge(reference_chunk, how='cross', suffixes=('_data', '_ref'))

    for col_name in cols_to_keep:
        sdtype = metadata['columns'][col_name]['sdtype']
        ref_column = full_dataset[col_name + '_ref']
        data_column = full_dataset[col_name + '_data']
        diff_col_name = col_name + '_diff'
        if sdtype in ['numerical', 'datetime']:
            diff = (ref_column - data_column).abs()
            if pd.api.types.is_timedelta64_dtype(diff):
                diff = diff.dt.total_seconds()

            full_dataset[col_name + '_diff'] = np.where(
                ranges[col_name] == 0,
                (diff > 0).astype(int),
                np.minimum(diff / ranges[col_name], 1.0),
            )

            xor_condition = (ref_column.isna() & ~data_column.isna()) | (
                ~ref_column.isna() & data_column.isna()
            )

            full_dataset.loc[xor_condition, diff_col_name] = 1

            both_nan_condition = ref_column.isna() & data_column.isna()

            full_dataset.loc[both_nan_condition, diff_col_name] = 0

        elif sdtype in ['categorical', 'boolean']:
            equals_cat = (ref_column == data_column) | (ref_column.isna() & data_column.isna())
            full_dataset[diff_col_name] = (~equals_cat).astype(int)

        full_dataset = full_dataset.drop(columns=[col_name + '_ref', col_name + '_data'])

    full_dataset['diff'] = full_dataset.iloc[:, 2:].sum(axis=1) / len(cols_to_keep)
    chunk_result = (
        full_dataset[['index_data', 'diff']].groupby('index_data').min().reset_index(drop=True)
    )
    return chunk_result['diff']


def calculate_dcr(dataset, reference_dataset, metadata, chunk_size=1000):
    """Calculate the Distance to Closest Record for all rows in the synthetic data.

    Arguments:
        dataset (pandas.Dataframe):
            The dataset for which we want to compute the DCR values
        reference_dataset (pandas.Dataframe):
            The reference dataset that is used for the distance computations
        metadata (dict):
            The metadata dict.

    Returns:
        pandas.Series:
            Returns a Series that shows the DCR value for every row of dataset
    """
    dataset = _process_data_with_metadata(dataset.copy(), metadata, True)
    reference = _process_data_with_metadata(reference_dataset.copy(), metadata, True)

    common_cols = set(dataset.columns) & set(reference.columns)
    cols_to_keep = []
    ranges = {}

    for col_name, col_metadata in get_columns_from_metadata(metadata).items():
        sdtype = col_metadata['sdtype']

        if (
            sdtype in ['numerical', 'categorical', 'boolean', 'datetime']
            and col_name in common_cols
        ):
            cols_to_keep.append(col_name)

            if sdtype in ['numerical', 'datetime']:
                col_range = reference[col_name].max() - reference[col_name].min()
                if isinstance(col_range, pd.Timedelta):
                    col_range = col_range.total_seconds()

                ranges[col_name] = col_range

    if not cols_to_keep:
        raise ValueError('There are no overlapping statistical columns to measure.')

    dataset = dataset[cols_to_keep]
    dataset['index'] = range(len(dataset))

    reference = reference[cols_to_keep]
    reference['index'] = range(len(reference))
    results = []

    for dataset_chunk_start in range(0, len(dataset), chunk_size):
        dataset_chunk = dataset.iloc[dataset_chunk_start : dataset_chunk_start + chunk_size]
        minimum_chunk_distance = None
        for reference_chunk_start in range(0, len(reference), chunk_size):
            reference_chunk = reference.iloc[
                reference_chunk_start : reference_chunk_start + chunk_size
            ]
            chunk_result = _process_dcr_chunk(
                dataset_chunk=dataset_chunk,
                reference_chunk=reference_chunk,
                cols_to_keep=cols_to_keep,
                metadata=metadata,
                ranges=ranges,
            )
            if minimum_chunk_distance is None:
                minimum_chunk_distance = chunk_result
            else:
                minimum_chunk_distance = pd.Series.min(
                    pd.concat([minimum_chunk_distance, chunk_result], axis=1), axis=1
                )

        results.append(minimum_chunk_distance)

    result = pd.concat(results, ignore_index=True)
    result.name = None

    return result
