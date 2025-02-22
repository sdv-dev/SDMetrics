"""Distance to closest record measurement functions."""

import pandas as pd

from sdmetrics.utils import is_datetime


def _calculate_dcr_value(s_value, d_value, sdtype, col_range=None):
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

    Returns:
        pandas.Dataframe:
            Returns a dataframe that shows the DCR value for all synthetic data.
    """
    if sdtype == 'numerical' or sdtype == 'datetime':
        if col_range is None:
            raise ValueError(
                'No col_range was provided. The col_range is required '
                'for numerical and datetime sdtype DCR calculation.'
            )
        if pd.isna(s_value) and pd.isna(d_value):
            return 0.0
        elif pd.isna(s_value) or pd.isna(d_value):
            return 1.0
        else:
            distance = abs(s_value - d_value) / (col_range)
            return min(distance, 1.0)
    else:
        if s_value == d_value:
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
        lambda s_col: _calculate_dcr_value(
            synthetic_row[s_col],
            comparison_row[s_col],
            metadata['columns'][s_col]['sdtype'],
            column_ranges.get(s_col),
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


def _covert_datetime_cols_unix_timestamp(data, metadata):
    for column in data.columns:
        if column in metadata['columns']:
            sdtype = metadata['columns'][column]['sdtype']
            if sdtype == 'datetime' and not is_datetime(data[column]):
                datetime_format = metadata['columns'][column].get('datetime_format')
                datetime_to_timestamp_col = pd.to_datetime(
                    data[column], format=datetime_format, errors='coerce'
                ).apply(_to_unix_timestamp)
                data[column] = datetime_to_timestamp_col


def calculate_dcr(synthetic_data, real_data, metadata):
    """Calculate the Distance to Closest Record for all rows in the synthetic data.

    Arguments:
        synthetic_data (pandas.Dataframe):
            The synthetic data that we are calculating DCR values for. Every row will be measured
            against the comparison data.
        real_data (pandas.Dataframe):
            The dataset that acts as the reference for DCR calculations. Ranges are determined from
            this dataset.
        metadata (dict):
            The metadata dict.

    Returns:
        pandas.Dataframe:
            Returns a dataframe that shows the DCR value for all synthetic data.
    """
    column_ranges = {}
    missing_cols = set(synthetic_data.index) - set(real_data.index)
    if missing_cols:
        raise ValueError(f'Different columns detected: {missing_cols}')

    r_data = real_data.copy()
    s_data = synthetic_data.copy()
    _covert_datetime_cols_unix_timestamp(r_data, metadata)
    _covert_datetime_cols_unix_timestamp(s_data, metadata)

    for column in r_data.columns:
        if column not in metadata['columns']:
            raise ValueError(f'Column {column} was not found in the metadata.')

        sdtype = metadata['columns'][column]['sdtype']
        col_range = None
        if sdtype == 'numerical' or sdtype == 'datetime':
            col_range = r_data[column].max() - r_data[column].min()

        column_ranges[column] = col_range

    dcr_dist_df = s_data.apply(
        lambda synth_row: _calculate_dcr_between_row_and_data(
            synth_row, r_data, column_ranges, metadata
        ),
        axis=1,
    )

    return dcr_dist_df
