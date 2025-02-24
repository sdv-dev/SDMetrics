"""Distance to closest record measurement functions."""

import warnings

import pandas as pd

from sdmetrics.utils import is_datetime


def _calculate_dcr_value(synthetic_value, real_value, sdtype, col_range=None):
    """Calculate the Distance to Closest Record between two different values.

    Arguments:
        s_value (int, float, datetime, boolean, string, or None):
            The synthetic value that we are calculating DCR value for
        d_value (int, float, datetime, boolean, string, or None):
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

        distance = abs(synthetic_value - real_value) / (col_range)
        return min(distance, 1.0)

    else:
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
        lambda synthetic_column_nam: _calculate_dcr_value(
            synthetic_row[synthetic_column_nam],
            comparison_row[synthetic_column_nam],
            metadata['columns'][synthetic_column_nam]['sdtype'],
            column_ranges.get(synthetic_column_nam),
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
    dist_srow_to_all_rows = real_data.apply(
        lambda d_row_obj: _calculate_dcr_between_rows(
            synthetic_row, d_row_obj, column_ranges, metadata
        ),
        axis=1,
    )
    return dist_srow_to_all_rows.min()


def _to_unix_timestamp(datetime_value):
    if not is_datetime(datetime_value):
        raise ValueError('Value is not of type pandas datetime.')
    return datetime_value.timestamp() if pd.notna(datetime_value) else pd.NaT


def _convert_datetime_cols_unix_timestamp_seconds(data, metadata):
    for column in data.columns:
        if column in metadata['columns']:
            sdtype = metadata['columns'][column]['sdtype']
            if sdtype == 'datetime' and not isinstance(data[column], float):
                datetime_format = metadata['columns'][column].get('datetime_format')
                if not datetime_format:
                    warnings.warn('No datetime format was specified.')
                datetime_to_timestamp_col = pd.to_datetime(
                    data[column], format=datetime_format, errors='coerce'
                ).apply(_to_unix_timestamp)
                data[column] = datetime_to_timestamp_col


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
        pandas.Dataframe:
            Returns a dataframe that shows the DCR value for all synthetic data.
    """
    column_ranges = {}
    missing_cols = set(real_data.columns) - set(synthetic_data.columns)
    if missing_cols:
        raise ValueError(f'Different columns detected: {missing_cols}')

    real_data_copy = real_data.copy()
    synthetic_data_copy = synthetic_data.copy()
    _convert_datetime_cols_unix_timestamp_seconds(real_data_copy, metadata)
    _convert_datetime_cols_unix_timestamp_seconds(synthetic_data_copy, metadata)

    for column in real_data_copy.columns:
        if column not in metadata['columns']:
            raise ValueError(f'Column {column} was not found in the metadata.')

        sdtype = metadata['columns'][column]['sdtype']
        col_range = None
        if sdtype == 'numerical' or sdtype == 'datetime':
            col_range = real_data_copy[column].max() - real_data_copy[column].min()

        column_ranges[column] = col_range

    dcr_dist_df = synthetic_data_copy.apply(
        lambda synth_row: _calculate_dcr_between_row_and_data(
            synth_row, real_data_copy, column_ranges, metadata
        ),
        axis=1,
    )

    return dcr_dist_df
