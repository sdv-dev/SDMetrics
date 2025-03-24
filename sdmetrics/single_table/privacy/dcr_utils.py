"""Distance to closest record measurement functions."""

import numpy as np
import pandas as pd

from sdmetrics._utils_metadata import _process_data_with_metadata
from sdmetrics.utils import get_columns_from_metadata


def calculate_dcr(dataset, reference_dataset, metadata):
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
    dataset_copy = _process_data_with_metadata(dataset.copy(), metadata, True)
    reference_copy = _process_data_with_metadata(reference_dataset.copy(), metadata, True)

    common_cols = set(dataset_copy.columns) & set(reference_copy.columns)
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
                col_range = reference_copy[col_name].max() - reference_copy[col_name].min()
                if isinstance(col_range, pd.Timedelta):
                    col_range = col_range.total_seconds()
                ranges[col_name] = col_range

    if not cols_to_keep:
        raise ValueError('There are no overlapping statistical columns to measure.')

    # perform a full cross join on the data we want to compute
    dataset_copy = dataset_copy[cols_to_keep]
    dataset_copy['index'] = [i for i in range(len(dataset_copy))]

    reference_copy = reference_copy[cols_to_keep]
    reference_copy['index'] = [i for i in range(len(reference_copy))]

    results = []
    chunk_size = 1000

    for chunk_start in range(0, len(dataset_copy), chunk_size):
        chunk = dataset_copy.iloc[chunk_start : chunk_start + chunk_size].copy()
        chunk['index'] = range(chunk_start, chunk_start + len(chunk))

        reference_copy['index'] = range(len(reference_copy))
        full_dataset = chunk.merge(reference_copy, how='cross', suffixes=('_data', '_ref'))

        for col_name in cols_to_keep:
            sdtype = metadata['columns'][col_name]['sdtype']
            if sdtype in ['numerical', 'datetime']:
                diff = (full_dataset[col_name + '_ref'] - full_dataset[col_name + '_data']).abs()
                if pd.api.types.is_timedelta64_dtype(diff):
                    diff = diff.dt.total_seconds()

                full_dataset[col_name + '_diff'] = np.where(
                    ranges[col_name] == 0,
                    (diff > 0).astype(int),
                    np.minimum(diff / ranges[col_name], 1.0),
                )

                xor_condition = (
                    full_dataset[col_name + '_ref'].isna()
                    & ~full_dataset[col_name + '_data'].isna()
                ) | (
                    ~full_dataset[col_name + '_ref'].isna()
                    & full_dataset[col_name + '_data'].isna()
                )

                full_dataset.loc[xor_condition, col_name + '_diff'] = 1

                both_nan_condition = (
                    full_dataset[col_name + '_ref'].isna() & full_dataset[col_name + '_data'].isna()
                )

                full_dataset.loc[both_nan_condition, col_name + '_diff'] = 0

            elif sdtype in ['categorical', 'boolean']:
                equals_cat = (
                    full_dataset[col_name + '_ref'] == full_dataset[col_name + '_data']
                ) | (
                    full_dataset[col_name + '_ref'].isna() & full_dataset[col_name + '_data'].isna()
                )
                full_dataset[col_name + '_diff'] = (~equals_cat).astype(int)

            full_dataset.drop(columns=[col_name + '_ref', col_name + '_data'], inplace=True)

        full_dataset['diff'] = full_dataset.iloc[:, 2:].sum(axis=1) / len(cols_to_keep)
        chunk_result = (
            full_dataset[['index_data', 'diff']].groupby('index_data').min().reset_index(drop=True)
        )
        results.append(chunk_result['diff'])

    result = pd.concat(results, ignore_index=True)
    result.name = None
    return result
