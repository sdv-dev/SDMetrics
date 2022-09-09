"""Report utility methods."""

import copy
import itertools

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.figure_factory as ff

from sdmetrics.utils import is_datetime

DATACEBO_DARK = '#000036'
DATACEBO_LIGHT = '#01E0C9'


def make_discrete_column_plot(real_column, synthetic_column, sdtype):
    """Plot the real and synthetic data for a categorical or boolean column.

    Args:
        real_column (pandas.Series):
            The real data for the desired column.
        synthetic_column (pandas.Series):
            The synthetic data for the desired column.
        sdtype (str):
            The data type of the column.

    Returns:
        plotly.graph_objects._figure.Figure
    """
    column_name = real_column.name if hasattr(real_column, 'name') else ''

    real_data = pd.DataFrame({'values': real_column.copy()})
    real_data['Data'] = 'Real'

    synthetic_data = pd.DataFrame({'values': synthetic_column.copy()})
    synthetic_data['Data'] = 'Synthetic'

    all_data = pd.concat([real_data, synthetic_data], axis=0, ignore_index=True)
    all_data = all_data.fillna('NaN')

    fig = px.histogram(
        all_data,
        x='values',
        color='Data',
        barmode='group',
        color_discrete_sequence=[DATACEBO_DARK, DATACEBO_LIGHT],
        pattern_shape='Data',
        pattern_shape_sequence=['', '/'],
        histnorm='probability density'
    )

    fig.update_traces(
        hovertemplate='<b>Real</b><br>Frequency: %{y}<extra></extra>',
        selector={'name': 'Real'}
    )

    fig.update_traces(
        hovertemplate='<b>Synthetic</b><br>Frequency: %{y}<extra></extra>',
        selector={'name': 'Synthetic'}
    )

    fig.update_layout(
        title=f"Real vs. Synthetic Data for column '{column_name}'",
        xaxis_title='Category',
        yaxis_title='Frequency',
        plot_bgcolor='#F5F5F8',
    )

    return fig


def make_continuous_column_plot(real_column, synthetic_column, sdtype):
    """Plot the real and synthetic data for a numerical or datetime column.

    Args:
        real_column (pandas.Series):
            The real data for the desired column.
        synthetic_column (pandas.Series):
            The synthetic data for the desired column.
        sdtype (str):
            The data type of the column.

    Returns:
        plotly.graph_objects._figure.Figure
    """
    column_name = real_column.name if hasattr(real_column, 'name') else ''
    missing_data_real = round((real_column.isna().sum() / len(real_column)) * 100, 2)
    missing_data_synthetic = round((synthetic_column.isna().sum() / len(synthetic_column)), 2)

    real_data = real_column.dropna()
    synthetic_data = synthetic_column.dropna()

    if sdtype == 'datetime':
        real_data = real_data.astype('int64')
        synthetic_data = synthetic_data.astype('int64')

    fig = ff.create_distplot(
        [real_data, synthetic_data],
        ['Real', 'Synthetic'],
        show_hist=False,
        show_rug=False,
        colors=[DATACEBO_DARK, DATACEBO_LIGHT]
    )

    fig.update_traces(
        x=pd.to_datetime(fig.data[0].x) if sdtype == 'datetime' else fig.data[0].x,
        fill='tozeroy',
        hovertemplate='<b>Real</b><br>Value: %{x}<br>Frequency: %{y}<extra></extra>',
        selector={'name': 'Real'}
    )

    fig.update_traces(
        x=pd.to_datetime(fig.data[1].x) if sdtype == 'datetime' else fig.data[1].x,
        fill='tozeroy',
        hovertemplate='<b>Synthetic</b><br>Value: %{x}<br>Frequency: %{y}<extra></extra>',
        selector={'name': 'Synthetic'}
    )

    fig.update_layout(
        title=f'Real vs. Synthetic Data for column {column_name}',
        xaxis_title='Value',
        yaxis_title='Frequency',
        plot_bgcolor='#F5F5F8',
        annotations=[
            {
                'xref': 'paper',
                'yref': 'paper',
                'x': -0.08,
                'y': -0.2,
                'showarrow': False,
                'text': (
                    f'*Missing Values: Real Data ({missing_data_real}%), '
                    f'Synthetic Data ({missing_data_synthetic}%)'
                ),
            },
        ]
    )

    return fig


def plot_column(real_column, synthetic_column, sdtype):
    """Plot the real and synthetic data for a given column.

    Args:
        real_column (pandas.Series):
            The real data for the desired column.
        synthetic_column (pandas.Series):
            The synthetic data for the desired column.
        sdtype (str):
            The data type of the column.
    """
    if sdtype == 'numerical' or sdtype == 'datetime':
        fig = make_continuous_column_plot(real_column, synthetic_column, sdtype)
    elif sdtype == 'categorical' or sdtype == 'boolean':
        fig = make_discrete_column_plot(real_column, synthetic_column, sdtype)

    fig.show()


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

    for field_name, field_meta in metadata['fields'].items():
        if field_meta['type'] == 'id':
            continue

        if field_meta['type'] == 'numerical' or field_meta['type'] == 'datetime':
            real_col = real_data[field_name]
            synthetic_col = synthetic_data[field_name]
            if is_datetime(real_col):
                real_col = pd.to_numeric(real_col)
                synthetic_col = pd.to_numeric(synthetic_col)

            bin_edges = np.histogram_bin_edges(real_col.dropna())
            binned_real_col = np.digitize(real_col, bins=bin_edges)
            binned_synthetic_col = np.digitize(synthetic_col, bins=bin_edges)

            binned_real[field_name] = binned_real_col
            binned_synthetic[field_name] = binned_synthetic_col
            binned_metadata['fields'][field_name] = {'type': 'categorical'}

    return binned_real, binned_synthetic, binned_metadata


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
        existing_keys (list[tuple(str)] or None):
            A list of keys for which to skip computing the metric.

    Returns:
        dict:
            The metric results.
    """
    metric_results = {}

    binned_real, binned_synthetic, binned_metadata = discretize_table_data(
        real_data, synthetic_data, metadata)

    non_id_cols = [
        field for field, field_meta in binned_metadata['fields'].items() if
        field_meta['type'] != 'id'
    ]
    for columns in itertools.combinations(non_id_cols, r=2):
        if columns not in keys_to_skip and (columns[1], columns[0]) not in keys_to_skip:
            result = metric.column_pairs_metric.compute_breakdown(
                binned_real[list(columns)],
                binned_synthetic[list(columns)],
            )
            metric_results[columns] = result
            metric_results[columns] = result

    return metric_results
