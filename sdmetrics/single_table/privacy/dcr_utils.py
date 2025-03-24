"""Distance to closest record measurement functions."""

import numpy as np
import pandas as pd

from sdmetrics._utils_metadata import _process_data_with_metadata
from sdmetrics.utils import get_columns_from_metadata


def _calculate_dcr_value(synthetic_value, real_value, sdtype, col_range=None):
    """Calculate the Distance to Closest Record between two different values.

    Arguments:
        synthetic_value (int, float, datetime, boolean, string, or None):
            The synthetic value that we are calculating DCR value for
        real_value (int, float, datetime, boolean, string, or None):
            The data value that we are referencing for measuring DCR.
        sdtype (string):
            The sdtype of the column values.
        col_range (float):
            The range of values for a column used for numerical values to calculate DCR.
            Defaults to None.

    Returns:
       float:
            Returns dcr value between two given values.
    """
    if pd.isna(synthetic_value) and pd.isna(real_value):
        return 0.0
    elif pd.isna(synthetic_value) or pd.isna(real_value):
        return 1.0

    if sdtype == 'numerical' or sdtype == 'datetime':
        if col_range is None:
            raise ValueError(
                'No col_range was provided. The col_range is required '
                'for numerical and datetime sdtype DCR calculation.'
            )

        difference = abs(synthetic_value - real_value)
        if isinstance(difference, pd.Timedelta):
            difference = difference.total_seconds()

        distance = 0.0 if synthetic_value == real_value else 1.0
        if col_range != 0:
            distance = difference / col_range

        return min(distance, 1.0)

    if synthetic_value == real_value:
        return 0.0
    else:
        return 1.0


def _calculate_dcr_between_rows(synthetic_row, comparison_row, column_ranges, metadata):
    """Calculate the Distance to Closest Record between two rows.

    Arguments:
        synthetic_row (pandas.Series):
            The synthetic row that we are calculating DCR value for.
        comparison_row (pandas.Series):
            The data value that we are referencing for measuring DCR.
        column_ranges (dict):
            A dictionary that defines the range for each numerical column.
        metadata (dict):
            The metadata dict.

    Returns:
        float:
            Returns DCR value (the average value of DCR values we computed across the row).
    """
    dcr_values = synthetic_row.index.to_series().apply(
        lambda synthetic_column_name: _calculate_dcr_value(
            synthetic_row[synthetic_column_name],
            comparison_row[synthetic_column_name],
            metadata['columns'][synthetic_column_name]['sdtype'],
            column_ranges.get(synthetic_column_name),
        )
    )

    return dcr_values.mean()


def _calculate_dcr_between_row_and_data(synthetic_row, real_data, column_ranges, metadata):
    """Calculate the DCR between a single row in the synthetic data and another dataset.

    Arguments:
        synthetic_row (pandas.Series):
            The synthetic row that we are calculating DCR against an entire dataset.
        real_data (pandas.Dataframe):
            The dataset that acts as the reference for DCR calculations.
        column_ranges (dict):
            A dictionary that defines the range for each numerical column.
        metadata (dict):
            The metadata dict.

    Returns:
        float:
            Returns the minimum distance to closest record computed between the
            synthetic row and the reference dataset.
    """
    synthetic_distance_to_all_real = real_data.apply(
        lambda real_row: _calculate_dcr_between_rows(
            synthetic_row, real_row, column_ranges, metadata
        ),
        axis=1,
    )
    return synthetic_distance_to_all_real.min()


def calculate_dcr(real_data, synthetic_data, metadata):
    """Calculate the Distance to Closest Record for all rows in the synthetic data.

    Arguments:
        real_data (pandas.Dataframe):
            The dataset that acts as the reference for DCR calculations. Ranges are determined from
            this dataset.
        synthetic_data (pandas.Dataframe):
            The synthetic data that we are calculating DCR values for. Every row will be measured
            against the comparison data.
        metadata (dict):
            The metadata dict.

    Returns:
        pandas.Series:
            Returns a Series that shows the DCR value for every row of synthetic data.
    """
    column_ranges = {}

    real_data_copy = real_data.copy()
    synthetic_data_copy = synthetic_data.copy()
    real_data_copy = _process_data_with_metadata(real_data_copy, metadata, True)
    synthetic_data_copy = _process_data_with_metadata(synthetic_data_copy, metadata, True)

    overlapping_columns = set(real_data_copy.columns) & set(synthetic_data_copy.columns)
    if not overlapping_columns:
        raise ValueError('There are no overlapping statistical columns to measure.')

    for col_name, column in get_columns_from_metadata(metadata).items():
        sdtype = column['sdtype']
        col_range = None
        if sdtype == 'numerical' or sdtype == 'datetime':
            col_range = real_data_copy[col_name].max() - real_data_copy[col_name].min()
            if isinstance(col_range, pd.Timedelta):
                col_range = col_range.total_seconds()

        column_ranges[col_name] = col_range

    dcr_dist_df = synthetic_data_copy.apply(
        lambda synth_row: _calculate_dcr_between_row_and_data(
            synth_row, real_data_copy, column_ranges, metadata
        ),
        axis=1,
    )

    return dcr_dist_df


def calculate_dcr_optimized(dataset, reference_dataset, metadata):
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

    # figure out which columns we want to keep for the computation
    # for this proof-of-concept, I've only implemented it on numerical and categorical
    # for the numerical columns, calculate the range based on the reference data
    cols_to_keep = []
    ranges = {}

    for col_name, col_metadata in get_columns_from_metadata(metadata).items():
        sdtype = col_metadata['sdtype']

        if sdtype in ['numerical', 'categorical', 'boolean', 'datetime']:
            cols_to_keep.append(col_name)

            if sdtype in ['numerical', 'datetime']:
                col_range = reference_copy[col_name].max() - reference_copy[col_name].min()
                if isinstance(col_range, pd.Timedelta):
                    col_range = col_range.total_seconds()
                ranges[col_name] = col_range

    # perform a full cross join on the data we want to compute
    dataset_copy = dataset_copy[cols_to_keep]
    dataset_copy['index'] = [i for i in range(len(dataset_copy))]

    reference_copy = reference_copy[cols_to_keep]
    reference_copy['index'] = [i for i in range(len(reference_copy))]

    full_dataset = dataset_copy.merge(reference_copy, how='cross', suffixes=('_data', '_ref'))

    # on the full dataset, we can now perform column-wise operations to compute differences
    # these are vectorized so they will be much faster
    print(cols_to_keep)
    for col_name in cols_to_keep:
        sdtype = metadata['columns'][col_name]['sdtype']
        if sdtype == 'numerical' or sdtype == 'datetime':
            diff = (full_dataset[col_name+'_ref'] - full_dataset[col_name+'_data']).abs()
            if (ranges[col_name] == 0):
                print(f'Column Range is 0 for : {col_name}')
            if isinstance(diff.iloc[0], pd.Timedelta):
                diff = diff.dt.total_seconds()

            full_dataset[col_name+'_diff'] = np.where(
                ranges[col_name] == 0,
                (diff > 0).astype(int),  # If values are different, assign 1; otherwise, 0
                np.minimum(diff / ranges[col_name], 1.0)  # Normalized difference when range > 0
            )
            print('Diff')
            print(diff)

        elif sdtype == 'categorical' or sdtype == 'boolean':
            equals_cat = ((full_dataset[col_name+'_ref'] == full_dataset[col_name+'_data']) |
                          (full_dataset[col_name+'_ref'].isna() & full_dataset[col_name+'_data'].isna()))
            full_dataset[col_name+'_diff'] = (~(equals_cat)).astype(int)

        full_dataset.drop(columns=[col_name+'_ref', col_name+'_data'], inplace=True)

    # the average distance is the overall distance
    full_dataset['diff'] = full_dataset.iloc[:, 2:].sum(axis=1) / len(cols_to_keep)

    # find the min distance for each of the data rows
    out = full_dataset[['index_data', 'diff']].groupby('index_data').min().reset_index(drop=True)
    return out['diff']
