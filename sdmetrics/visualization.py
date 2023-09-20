"""Visualization methods for SDMetrics."""

import pandas as pd
import plotly.express as px
import plotly.figure_factory as ff
from pandas.api.types import is_datetime64_dtype

from sdmetrics.reports.utils import PlotConfig
from sdmetrics.utils import get_missing_percentage


def _generate_column_bar_plot(real_data, synthetic_data, plot_kwargs={}):
    """Generate a bar plot of the real and synthetic data.

    Args:
        real_column (pandas.Series):
            The real data for the desired column.
        synthetic_column (pandas.Series):
            The synthetic data for the desired column.
        plot_kwargs (dict, optional):
            Dictionary of keyword arguments to pass to px.histogram. Keyword arguments
            provided this way will overwrite defaults.

    Returns:
        plotly.graph_objects._figure.Figure
    """
    all_data = pd.concat([real_data, synthetic_data], axis=0, ignore_index=True)
    default_histogram_kwargs = {
        'x': 'values',
        'color': 'Data',
        'barmode': 'group',
        'color_discrete_sequence': [PlotConfig.DATACEBO_DARK, PlotConfig.DATACEBO_GREEN],
        'pattern_shape': 'Data',
        'pattern_shape_sequence': ['', '/'],
        'histnorm': 'probability density',
    }
    fig = px.histogram(
        all_data,
        **{**default_histogram_kwargs, **plot_kwargs}
    )

    return fig


def _generate_column_distplot(real_data, synthetic_data, plot_kwargs={}):
    """Plot the real and synthetic data as a distplot.

    Args:
        real_data (pandas.DataFrame):
            The real data for the desired column.
        synthetic_data (pandas.DataFrame):
            The synthetic data for the desired column.
        plot_kwargs (dict, optional):
            Dictionary of keyword arguments to pass to px.histogram. Keyword arguments
            provided this way will overwrite defaults.

    Returns:
        plotly.graph_objects._figure.Figure
    """
    default_distplot_kwargs = {
        'show_hist': False,
        'show_rug': False,
        'colors': [PlotConfig.DATACEBO_DARK, PlotConfig.DATACEBO_GREEN]
    }

    fig = ff.create_distplot(
        [real_data['values'], synthetic_data['values']],
        ['Real', 'Synthetic'],
        **{**default_distplot_kwargs, **plot_kwargs}
    )

    return fig


def _generate_column_plot(real_column,
                          synthetic_column,
                          plot_type,
                          plot_kwargs={},
                          plot_title=None,
                          x_label=None):
    """Generate a plot of the real and synthetic data.

    Args:
        real_column (pandas.Series):
            The real data for the desired column.
        synthetic_column (pandas.Series):
            The synthetic data for the desired column.
        plot_type (str):
            The type of plot to use. Must be one of 'bar' or 'distplot'.
        hist_kwargs (dict, optional):
            Dictionary of keyword arguments to pass to px.histogram. Keyword arguments
            provided this way will overwrite defaults.
        plot_title (str, optional):
            Title to use for the plot. Defaults to 'Real vs. Synthetic Data for column {column}'
        x_label (str, optional):
            Label to use for x-axis. Defaults to 'Category'.

    Returns:
        plotly.graph_objects._figure.Figure
    """
    if plot_type not in ['bar', 'distplot']:
        raise ValueError(
            "Unrecognized plot_type '{plot_type}'. Pleas use one of 'bar' or 'distplot'"
        )

    column_name = real_column.name if hasattr(real_column, 'name') else ''

    real_data = pd.DataFrame({'values': real_column.copy()})
    real_data['Data'] = 'Real'
    synthetic_data = pd.DataFrame({'values': synthetic_column.copy()})
    synthetic_data['Data'] = 'Synthetic'

    is_datetime_sdtype = False
    if is_datetime64_dtype(real_column.dtype):
        is_datetime_sdtype = True
        real_data = real_data.astype('int64')
        synthetic_data = synthetic_data.astype('int64')

    missing_data_real = get_missing_percentage(real_column)
    missing_data_synthetic = get_missing_percentage(synthetic_column)

    trace_args = {}

    if plot_type == 'bar':
        fig = _generate_column_bar_plot(real_data, synthetic_data, plot_kwargs)
    elif plot_type == 'distplot':
        fig = _generate_column_distplot(real_data, synthetic_data, plot_kwargs)
        trace_args = {'fill': 'tozeroy'}

    for i, name in enumerate(['Real', 'Synthetic']):
        fig.update_traces(
            x=pd.to_datetime(fig.data[i].x) if is_datetime_sdtype else fig.data[i].x,
            hovertemplate=f'<b>{name}</b><br>Frequency: %{{y}}<extra></extra>',
            selector={'name': name},
            **trace_args
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

    if not plot_title:
        plot_title = f"Real vs. Synthetic Data for column '{column_name}'"

    if not x_label:
        x_label = 'Category'

    fig.update_layout(
        title=plot_title,
        xaxis_title=x_label,
        yaxis_title='Frequency',
        plot_bgcolor=PlotConfig.BACKGROUND_COLOR,
        annotations=annotations,
        font={'size': PlotConfig.FONT_SIZE},
    )
    return fig


def _generate_cardinality_plot(real_data,
                               synthetic_data,
                               parent_primary_key,
                               child_foreign_key,
                               plot_type='bar'):
    plot_title = (
        f"Relationship (child foreign key='{child_foreign_key}' and parent "
        f"primary key='{parent_primary_key}')"
    )
    x_label = '# of Children (per Parent)'

    plot_kwargs = {}
    if plot_type == 'bar':
        max_cardinality = max(max(real_data), max(synthetic_data))
        min_cardinality = min(min(real_data), min(synthetic_data))
        plot_kwargs = {
            'nbins': max_cardinality - min_cardinality + 1
        }

    return _generate_column_plot(real_data, synthetic_data, plot_type,
                                 plot_kwargs, plot_title, x_label)


def _get_cardinality(parent_table, child_table, parent_primary_key, child_foreign_key):
    """Return the cardinality of the parent-child relationship.

    Args:
        parent_table (pandas.DataFrame):
            The parent table.
        child_table (pandas.DataFrame):
            The child table.
        parent_primary_key (string):
            The name of the primary key column in the parent table.
        child_foreign_key (string):
            The name of the foreign key column in the child table.

    Returns:
        pandas.DataFrame
    """
    child_counts = child_table[child_foreign_key].value_counts().rename('# children')
    cardinalities = child_counts.reindex(parent_table[parent_primary_key], fill_value=0).to_frame()

    return cardinalities.sort_values('# children')['# children']


def get_cardinality_plot(real_data, synthetic_data, child_table_name, parent_table_name,
                         child_foreign_key, parent_primary_key, plot_type='bar'):
    """Return a plot of the cardinality of the parent-child relationship.

    Args:
        real_data (dict):
            The real data.
        synthetic_data (dict):
            The synthetic data.
        child_table_name (string):
            The name of the child table.
        parent_table_name (string):
            The name of the parent table.
        child_foreign_key (string):
            The name of the foreign key column in the child table.
        parent_primary_key (string):
            The name of the primary key column in the parent table.
        plot_type (string, optional):
            The plot type to use to plot the cardinality. Must be either 'bar' or 'distplot'.
            Defaults to 'bar'.

    Returns:
        plotly.graph_objects._figure.Figure
    """
    if plot_type not in ['bar', 'distplot']:
        raise ValueError(
            f"Invalid plot_type '{plot_type}'. Please use one of ['bar', 'distplot'].")

    real_cardinality = _get_cardinality(
        real_data[parent_table_name], real_data[child_table_name],
        parent_primary_key, child_foreign_key
    )
    synth_cardinality = _get_cardinality(
        synthetic_data[parent_table_name],
        synthetic_data[child_table_name],
        parent_primary_key, child_foreign_key
    )

    fig = _generate_cardinality_plot(
        real_cardinality,
        synth_cardinality,
        parent_primary_key,
        child_foreign_key,
        plot_type=plot_type
    )

    return fig
