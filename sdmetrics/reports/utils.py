"""Report utility methods."""

import copy
import itertools

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.figure_factory as ff
from pandas.core.tools.datetimes import _guess_datetime_format_for_array

from sdmetrics.utils import is_datetime

DATACEBO_DARK = '#000036'
DATACEBO_LIGHT = '#01E0C9'
BACKGROUND_COLOR = '#F5F5F8'
CONTINUOUS_SDTYPES = ['numerical', 'datetime']
DISCRETE_SDTYPES = ['categorical', 'boolean']


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

    missing_data_real = round((real_column.isna().sum() / len(real_column)) * 100, 2)
    missing_data_synthetic = round((synthetic_column.isna().sum() / len(synthetic_column)), 2)

    all_data = pd.concat([real_data, synthetic_data], axis=0, ignore_index=True)

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

    show_missing_values = missing_data_real > 0 or missing_data_synthetic > 0

    annotations = [] if not show_missing_values else [
        {
            'xref': 'paper',
            'yref': 'paper',
            'x': 1.0,
            'y': 1.05,
            'showarrow': False,
            'text': (
                f'*Missing Values: Real Data ({missing_data_real}%), '
                f'Synthetic Data ({missing_data_synthetic}%)'
            ),
        },
    ]

    fig.update_layout(
        title=f"Real vs. Synthetic Data for column '{column_name}'",
        xaxis_title='Category',
        yaxis_title='Frequency',
        plot_bgcolor=BACKGROUND_COLOR,
        annotations=annotations,
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

    show_missing_values = missing_data_real > 0 or missing_data_synthetic > 0

    annotations = [] if not show_missing_values else [
        {
            'xref': 'paper',
            'yref': 'paper',
            'x': 1.0,
            'y': 1.05,
            'showarrow': False,
            'text': (
                f'*Missing Values: Real Data ({missing_data_real}%), '
                f'Synthetic Data ({missing_data_synthetic}%)'
            ),
        },
    ]

    fig.update_layout(
        title=f'Real vs. Synthetic Data for column {column_name}',
        xaxis_title='Value',
        yaxis_title='Frequency',
        plot_bgcolor=BACKGROUND_COLOR,
        annotations=annotations,
    )

    return fig


def get_column_plot(real_data, synthetic_data, column_name, metadata):
    """Return a plot of the real and synthetic data for a given column.

    Args:
        real_data (pandas.DataFrame):
            The real table data.
        synthetic_data (pandas.DataFrame):
            The synthetic table data.
        column_name (str):
            The name of the column.
        metadata (dict):
            The table metadata.

    Returns:
        plotly.graph_objects._figure.Figure
    """
    if column_name not in metadata['fields']:
        raise ValueError(f"Column '{column_name}' not found in metadata.")
    elif 'type' not in metadata['fields'][column_name]:
        raise ValueError(f"Metadata for column '{column_name}' missing 'type' information.")
    if column_name not in real_data.columns:
        raise ValueError(f"Column '{column_name}' not found in real table data.")
    if column_name not in synthetic_data.columns:
        raise ValueError(f"Column '{column_name}' not found in synthetic table data.")

    sdtype = metadata['fields'][column_name]['type']
    real_column = real_data[column_name]
    synthetic_column = synthetic_data[column_name]
    if sdtype in CONTINUOUS_SDTYPES:
        fig = make_continuous_column_plot(real_column, synthetic_column, sdtype)
    elif sdtype in DISCRETE_SDTYPES:
        fig = make_discrete_column_plot(real_column, synthetic_column, sdtype)
    else:
        raise ValueError(f"sdtype of type '{sdtype}' not recognized.")

    return fig


def convert_to_datetime(column_data):
    """Convert a column data to pandas datetime.

    Args:
        column_data (pandas.Series):
            The column data

    Returns:
        pandas.Series:
            The converted column data.
    """
    if is_datetime(column_data):
        return column_data

    dt_format = _guess_datetime_format_for_array(column_data.astype(str).to_numpy())
    return pd.to_datetime(column_data, format=dt_format)


def make_continuous_column_pair_plot(real_data, synthetic_data):
    """Make a column pair plot for continuous data.

    Args:
        real_data (pandas.DataFrame):
            The real data for the desired column pair.
        synthetic_column (pandas.Dataframe):
            The synthetic data for the desired column pair.

    Returns:
        plotly.graph_objects._figure.Figure
    """
    columns = real_data.columns
    real_data = real_data.copy()
    real_data['Data'] = 'Real'
    synthetic_data = synthetic_data.copy()
    synthetic_data['Data'] = 'Synthetic'
    all_data = pd.concat([real_data, synthetic_data], axis=0, ignore_index=True)

    fig = px.scatter(
        all_data,
        x=columns[0],
        y=columns[1],
        color='Data',
        color_discrete_map={'Real': DATACEBO_DARK, 'Synthetic': DATACEBO_LIGHT},
        symbol='Data'
    )

    fig.update_layout(
        title=f"Real vs. Synthetic Data for columns '{columns[0]}' and '{columns[1]}'",
        plot_bgcolor=BACKGROUND_COLOR,
    )

    return fig


def make_discrete_column_pair_plot(real_data, synthetic_data):
    """Make a column pair plot for discrete data.

    Args:
        real_data (pandas.DataFrame):
            The real data for the desired column pair.
        synthetic_column (pandas.Dataframe):
            The synthetic data for the desired column pair.

    Returns:
        plotly.graph_objects._figure.Figure
    """
    columns = real_data.columns
    real_data = real_data.copy()
    real_data['Data'] = 'Real'
    synthetic_data = synthetic_data.copy()
    synthetic_data['Data'] = 'Synthetic'
    all_data = pd.concat([real_data, synthetic_data], axis=0, ignore_index=True)

    fig = px.density_heatmap(
        all_data,
        x=columns[0],
        y=columns[1],
        facet_col='Data',
        histnorm='probability'
    )

    fig.update_layout(
        title_text=f"Real vs Synthetic Data for columns '{columns[0]}' and '{columns[1]}'",
        coloraxis={'colorscale': [DATACEBO_DARK, DATACEBO_LIGHT]},
    )

    fig.for_each_annotation(lambda a: a.update(text=a.text.split('=')[-1] + ' Data'))

    return fig


def make_mixed_column_pair_plot(real_data, synthetic_data):
    """Make a column pair plot for mixed discrete and continuous column data.

    Args:
        real_data (pandas.DataFrame):
            The real data for the desired column pair.
        synthetic_column (pandas.Dataframe):
            The synthetic data for the desired column pair.

    Returns:
        plotly.graph_objects._figure.Figure
    """
    columns = real_data.columns
    real_data = real_data.copy()
    real_data['Data'] = 'Real'
    synthetic_data = synthetic_data.copy()
    synthetic_data['Data'] = 'Synthetic'
    all_data = pd.concat([real_data, synthetic_data], axis=0, ignore_index=True)

    fig = px.box(
        all_data,
        x=columns[0],
        y=columns[1],
        color='Data',
        color_discrete_map={'Real': DATACEBO_DARK, 'Synthetic': DATACEBO_LIGHT},
    )

    fig.update_layout(
        title=f"Real vs. Synthetic Data for columns '{columns[0]}' and '{columns[1]}'",
        plot_bgcolor=BACKGROUND_COLOR
    )

    return fig


def get_column_pair_plot(real_data, synthetic_data, columns, metadata):
    """Return a plot of the real and synthetic data for a given column pair.

    Args:
        real_data (pandas.DataFrame):
            The real table data.
        synthetic_column (pandas.Dataframe):
            The synthetic table data.
        columns (list[string]):
            The two columns to plot.
        metadata (dict):
            The table metadata.

    Returns:
        plotly.graph_objects._figure.Figure
    """
    invalid_columns = [column for column in columns if column not in metadata['fields']]
    if invalid_columns:
        raise ValueError(f"Column(s) `{'`, `'.join(invalid_columns)}` not found in metadata.")
    else:
        invalid_columns = [
            column for column in columns if 'type' not in metadata['fields'][column]
        ]
        if invalid_columns:
            raise ValueError(f"Metadata for column(s) `{'`, `'.join(invalid_columns)}` "
                             "missing 'type' information.")

    invalid_columns = [column for column in columns if column not in real_data.columns]
    if invalid_columns:
        raise ValueError(f"Column(s) `{'`, `'.join(invalid_columns)}` not found "
                         'in the real table data.')

    invalid_columns = [column for column in columns if column not in synthetic_data.columns]
    if invalid_columns:
        raise ValueError(f"Column(s) `{'`, `'.join(invalid_columns)}` not found "
                         'in the synthetic table data.')

    sdtypes = (metadata['fields'][columns[0]]['type'], metadata['fields'][columns[1]]['type'])
    real_data = real_data[columns]
    synthetic_data = synthetic_data[columns]

    all_sdtypes = CONTINUOUS_SDTYPES + DISCRETE_SDTYPES
    invalid_sdtypes = [sdtype for sdtype in sdtypes if sdtype not in all_sdtypes]
    if invalid_sdtypes:
        raise ValueError(f"sdtype(s) of type `{'`, `'.join(invalid_sdtypes)}` not recognized.")

    if all([t in DISCRETE_SDTYPES for t in sdtypes]):
        return make_discrete_column_pair_plot(real_data, synthetic_data)

    if sdtypes[0] == 'datetime':
        real_data.iloc[:, 0] = convert_to_datetime(real_data.iloc[:, 0])
    if sdtypes[1] == 'datetime':
        real_data.iloc[:, 1] = convert_to_datetime(real_data.iloc[:, 1])

    if all([t in CONTINUOUS_SDTYPES for t in sdtypes]):
        return make_continuous_column_pair_plot(real_data, synthetic_data)
    else:
        return make_mixed_column_pair_plot(real_data, synthetic_data)


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
            if field_meta['type'] == 'datetime':
                if real_col.dtype == 'O' and field_meta.get('format', ''):
                    real_col = pd.to_datetime(real_col, format=field_meta['format'])
                    synthetic_col = pd.to_datetime(synthetic_col, format=field_meta['format'])

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
        sorted_columns = tuple(sorted(columns))
        if (
            sorted_columns not in keys_to_skip and
            (sorted_columns[1], sorted_columns[0]) not in keys_to_skip
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
