"""Distance to closest record measurement functions."""

import pandas as pd

from sdmetrics.utils import is_datetime


def _calculate_dcr_value(s_value, d_value, col_name, metadata, range=None):
    """Calculate the Distance to Closest Record between two different values.

    Arguments:
        s_value (int, float, datetime, boolean, string, or None):
            The synthetic value that we are calculating DCR value for
        d_value (int, float, datetime, boolean, string, or None):
            The data value that we are referencing for measuring DCR.
        col_name (string):
            The column name in the metadata dictionary that will be used to
            determine how the type is measured.
        metadata (dict):
            The metadata dict.
        range (float):
            The range of values for a column used for numerical values to calculate DCR.

    Returns:
        pandas.Dataframe:
            Returns a dataframe that shows the DCR value for all synthetic data.
    """
    if col_name not in metadata['columns']:
        raise ValueError(f'Column {col_name} was not found in the metadata.')

    sdtype = metadata['columns'][col_name]['sdtype']
    if sdtype == 'numerical' or sdtype == 'datetime':
        if range is None:
            raise ValueError(
                f'The numerical column: {col_name} did not produce a range. '
                'Check that column has sdtype=numerical and that it exists in training data.'
            )
        if pd.isna(s_value) and pd.isna(d_value):
            return 0.0
        elif pd.isna(s_value) or pd.isna(d_value):
            return 1.0
        else:
            if sdtype == 'datetime':
                datetime_format = metadata['columns'][col_name].get('datetime_format')
                s_value = pd.to_datetime(s_value, format=datetime_format).timestamp()
                d_value = pd.to_datetime(d_value, format=datetime_format).timestamp()
            distance = abs(s_value - d_value) / (range)
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
    dcr_list_of_dist = []
    for s_col in synthetic_row.index:
        if s_col not in comparison_row.index:
            raise ValueError(
                f'Column name ({s_col}) was not found when calculating DCR between two rows.'
            )
        d_value = comparison_row.loc[s_col]
        s_value = synthetic_row.loc[s_col]
        dist = _calculate_dcr_value(s_value, d_value, s_col, metadata, column_ranges.get(s_col))
        dcr_list_of_dist.append(dist)

    return sum(dcr_list_of_dist) / len(dcr_list_of_dist)


def _calculate_dcr_between_row_and_data(synthetic_row, comparison_data, column_ranges, metadata):
    """Calculate the DCR between a single row in the synthetic data and another dataset.

    Arguments:
        synthetic_row (pandas.Series):
            The synthetic row that we are calculating DCR against an entire dataset.
        comparison_date (pandas.Dataframe):
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
    dist_srow_to_all_rows = comparison_data.apply(
        lambda d_row_obj: _calculate_dcr_between_rows(
            synthetic_row, d_row_obj, column_ranges, metadata
        ),
        axis=1,
    )
    return dist_srow_to_all_rows.min()


def calculate_dcr(synthetic_data, comparison_data, metadata):
    """Calculate the Distance to Closest Record for all rows in the synthetic data.

    Arguments:
        synthetic_data (pandas.Dataframe):
            The synthetic data that we are calculating DCR values for. Every row will be measured
            against the comparison data.
        comparison_date (pandas.Dataframe):
            The dataset that acts as the reference for DCR calculations. Ranges are determined from
            this dataset.
        metadata (dict):
            The metadata dict.

    Returns:
        pandas.Dataframe:
            Returns a dataframe that shows the DCR value for all synthetic data.
    """
    column_ranges = {}
    for column in comparison_data.columns:
        if column not in metadata['columns']:
            raise ValueError(f'Column {column} was not found in the metadata.')
        sdtype = metadata['columns'][column]['sdtype']
        if sdtype == 'numerical':
            col_range = comparison_data[column].max() - comparison_data[column].min()
        elif sdtype == 'datetime':
            datetime_format = metadata['columns'][column].get('datetime_format')
            datetime_to_timestamp_col = comparison_data[column]

            if not is_datetime(datetime_to_timestamp_col):
                datetime_to_timestamp_col = pd.to_datetime(
                    datetime_to_timestamp_col, format=datetime_format, errors='coerce'
                )

            if not isinstance(datetime_to_timestamp_col, pd.Timestamp):
                datetime_to_timestamp_col = datetime_to_timestamp_col.apply(
                    lambda x: x.timestamp() if pd.notna(x) else x
                )

            col_range = datetime_to_timestamp_col.max() - datetime_to_timestamp_col.min()
        column_ranges[column] = col_range

    dcr_dist_dict = synthetic_data.apply(
        lambda synth_row: _calculate_dcr_between_row_and_data(
            synth_row, comparison_data, column_ranges, metadata
        ),
        axis=1,
    )

    return dcr_dist_dict
