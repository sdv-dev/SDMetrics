"""Visualization methods for SDMetrics."""

from functools import wraps

import pandas as pd
import plotly.express as px
import plotly.figure_factory as ff
import plotly.graph_objects as go
import plotly.io as pio
from pandas.api.types import is_datetime64_dtype

from sdmetrics.reports.utils import PlotConfig
from sdmetrics.utils import get_missing_percentage, is_datetime


def set_plotly_config(function):
    """Set the ``plotly.io.renders`` config according to the environment.

    Configure the rendering settings based on the environment in which the plot is generated
    to ensure the image rendering with a stable engine. For other environments, like
    ``Jupyter Notebooks``, select the ``iframe`` rendering engine otherwise leave the default
    one.
    """
    @wraps(function)
    def wrapper(*args, **kwargs):
        renderers = list(pio.renderers)
        try:
            # Lazy import IPython
            from IPython import get_ipython

            ipython_interpreter = str(get_ipython())
            if 'ZMQInteractiveShell' in ipython_interpreter and 'iframe' in renderers:
                # This means we are using jupyter notebook
                pio.renderers.default = 'iframe'

        except Exception:
            pass

        return function(*args, **kwargs)

    return wrapper


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
    histogram_kwargs = {
        'x': 'values',
        'color': 'Data',
        'barmode': 'group',
        'color_discrete_sequence': [PlotConfig.DATACEBO_DARK, PlotConfig.DATACEBO_GREEN],
        'pattern_shape': 'Data',
        'pattern_shape_sequence': ['', '/'],
        'histnorm': 'probability density',
    }
    histogram_kwargs.update(plot_kwargs)
    fig = px.histogram(
        all_data,
        **histogram_kwargs
    )

    return fig


def _generate_heatmap_plot(all_data, columns):
    """Generate heatmap plot for discrete data.

    Args:
        all_data (pandas.DataFrame):
            The real and synthetic data for the desired column pair containing a
            ``Data`` column that indicates whether is real or synthetic.
        columns (list):
            A list of the columns being plotted.

    Returns:
        plotly.graph_objects._figure.Figure
    """
    fig = px.density_heatmap(
        all_data,
        x=columns[0],
        y=columns[1],
        facet_col='Data',
        histnorm='probability'
    )

    fig.update_layout(
        title_text=f"Real vs Synthetic Data for columns '{columns[0]}' and '{columns[1]}'",
        coloraxis={'colorscale': [PlotConfig.DATACEBO_DARK, PlotConfig.DATACEBO_GREEN]},
        font={'size': PlotConfig.FONT_SIZE},
    )

    fig.for_each_annotation(lambda a: a.update(text=a.text.split('=')[-1] + ' Data'))

    return fig


def _generate_box_plot(all_data, columns):
    """Generate a box plot for mixed discrete and continuous column data.

    Args:
        all_data (pandas.DataFrame):
            The real and synthetic data for the desired column pair containing a
            ``Data`` column that indicates whether is real or synthetic.
        columns (list):
            A list of the columns being plotted.

    Returns:
        plotly.graph_objects._figure.Figure
    """
    fig = px.box(
        all_data,
        x=columns[0],
        y=columns[1],
        color='Data',
        color_discrete_map={
            'Real': PlotConfig.DATACEBO_DARK,
            'Synthetic': PlotConfig.DATACEBO_GREEN
        },
    )

    fig.update_layout(
        title=f"Real vs. Synthetic Data for columns '{columns[0]}' and '{columns[1]}'",
        plot_bgcolor=PlotConfig.BACKGROUND_COLOR,
        font={'size': PlotConfig.FONT_SIZE},
    )

    return fig


def _generate_scatter_plot(all_data, columns):
    """Generate a scatter plot for column pair plot.

    Args:
        all_data (pandas.DataFrame):
            The real and synthetic data for the desired column pair containing a
            ``Data`` column that indicates whether is real or synthetic.
        columns (list):
            A list of the columns being plotted.

    Returns:
        plotly.graph_objects._figure.Figure
    """
    fig = px.scatter(
        all_data,
        x=columns[0],
        y=columns[1],
        color='Data',
        color_discrete_map={
            'Real': PlotConfig.DATACEBO_DARK,
            'Synthetic': PlotConfig.DATACEBO_GREEN
        },
        symbol='Data'
    )

    fig.update_layout(
        title=f"Real vs. Synthetic Data for columns '{columns[0]}' and '{columns[1]}'",
        plot_bgcolor=PlotConfig.BACKGROUND_COLOR,
        font={'size': PlotConfig.FONT_SIZE},
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

    missing_data_real = get_missing_percentage(real_column)
    missing_data_synthetic = get_missing_percentage(synthetic_column)

    real_data = pd.DataFrame({'values': real_column.copy().dropna()})
    real_data['Data'] = 'Real'
    synthetic_data = pd.DataFrame({'values': synthetic_column.copy().dropna()})
    synthetic_data['Data'] = 'Synthetic'

    is_datetime_sdtype = False
    if is_datetime64_dtype(real_column.dtype):
        is_datetime_sdtype = True
        real_data['values'] = real_data['values'].astype('int64')
        synthetic_data['values'] = synthetic_data['values'].astype('int64')

    trace_args = {}

    if plot_type == 'bar':
        fig = _generate_column_bar_plot(real_data, synthetic_data, plot_kwargs)
    elif plot_type == 'distplot':
        x_label = x_label or 'Value'
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


@set_plotly_config
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


@set_plotly_config
def get_column_plot(real_data, synthetic_data, column_name, plot_type=None):
    """Return a plot of the real and synthetic data for a given column.

    Args:
        real_data (pandas.DataFrame):
            The real table data.
        synthetic_data (pandas.DataFrame):
            The synthetic table data.
        column_name (str):
            The name of the column.
        plot_type (str or None):
            The plot to be used. Can choose between ``distplot``, ``bar`` or ``None``. If ``None`
            select between ``distplot`` or ``bar`` depending on the data that the column contains,
            ``distplot`` for datetime and numerical values and ``bar`` for categorical.
            Defaults to ``None``.

    Returns:
        plotly.graph_objects._figure.Figure
    """
    if plot_type not in ['bar', 'distplot', None]:
        raise ValueError(
            f"Invalid plot_type '{plot_type}'. Please use one of ['bar', 'distplot', None]."
        )

    if column_name not in real_data.columns:
        raise ValueError(f"Column '{column_name}' not found in real table data.")
    if column_name not in synthetic_data.columns:
        raise ValueError(f"Column '{column_name}' not found in synthetic table data.")

    real_column = real_data[column_name]
    if plot_type is None:
        column_is_datetime = is_datetime(real_data[column_name])
        dtype = real_column.dropna().infer_objects().dtype.kind
        if column_is_datetime or dtype in ('i', 'f'):
            plot_type = 'distplot'
        else:
            plot_type = 'bar'

    real_column = real_data[column_name]
    synthetic_column = synthetic_data[column_name]

    fig = _generate_column_plot(real_column, synthetic_column, plot_type)

    return fig


@set_plotly_config
def get_column_pair_plot(real_data, synthetic_data, column_names, plot_type=None):
    """Return a plot of the real and synthetic data for a given column pair.

    Args:
        real_data (pandas.DataFrame):
            The real table data.
        synthetic_column (pandas.Dataframe):
            The synthetic table data.
        column_names (list[string]):
            The names of the two columns to plot.
        plot_type (str or None):
            The plot to be used. Can choose between ``box``, ``heatmap``, ``scatter`` or ``None``.
            If ``None` select between ``box``, ``heatmap`` or ``scatter`` depending on the data
            that the column contains, ``scatter`` used for datetime and numerical values,
            ``heatmap`` for categorical and ``box`` for a mix of both. Defaults to ``None``.

    Returns:
        plotly.graph_objects._figure.Figure
    """
    if len(column_names) != 2:
        raise ValueError('Must provide exactly two column names.')

    if not set(column_names).issubset(real_data.columns):
        raise ValueError(
            f'Missing column(s) {set(column_names) - set(real_data.columns)} in real data.'
        )

    if not set(column_names).issubset(synthetic_data.columns):
        raise ValueError(
            f'Missing column(s) {set(column_names) - set(synthetic_data.columns)} '
            'in synthetic data.'
        )

    if plot_type not in ['box', 'heatmap', 'scatter', None]:
        raise ValueError(
            f"Invalid plot_type '{plot_type}'. Please use one of "
            "['box', 'heatmap', 'scatter', None]."
        )

    real_data = real_data[column_names]
    synthetic_data = synthetic_data[column_names]
    if plot_type is None:
        plot_type = []
        for column_name in column_names:
            column = real_data[column_name]
            dtype = column.dropna().infer_objects().dtype.kind
            if dtype in ('i', 'f') or is_datetime(column):
                plot_type.append('scatter')
            else:
                plot_type.append('heatmap')

        if len(set(plot_type)) > 1:
            plot_type = 'box'
        else:
            plot_type = plot_type.pop()

    # Merge the real and synthetic data and add a flag ``Data`` to indicate each one.
    columns = list(real_data.columns)
    real_data = real_data.copy()
    real_data['Data'] = 'Real'
    synthetic_data = synthetic_data.copy()
    synthetic_data['Data'] = 'Synthetic'
    all_data = pd.concat([real_data, synthetic_data], axis=0, ignore_index=True)

    if plot_type == 'scatter':
        return _generate_scatter_plot(all_data, columns)
    elif plot_type == 'heatmap':
        return _generate_heatmap_plot(all_data, columns)

    return _generate_box_plot(all_data, columns)


def _generate_line_plot(real_data, synthetic_data, x_axis, y_axis, marker, annotations=None):
    """Generate a line plot of the real and synthetic data separated by a marker column.

    Args:
        real_data (pandas.DataFrame):
            The real table data.
        synthetic_column (pandas.Dataframe):
            The synthetic table data.
        x_axis (str):
            The column name to be used as the x-axis of the graph
        y_axis (str):
            The column name to be used as the y-axis of the graph
        marker (str):
            The column used to define separate line sequences
        annotations (None or dict):
            Dict object that describes additional information to be presented in the graph

    Returns:
        plotly.graph_objects._figure.Figure
    """
    # Check if the column is the appropriate type
    all_data = pd.concat([real_data, synthetic_data], axis=0, ignore_index=True)
    if not (is_datetime(all_data[x_axis]) or
            pd.api.types.is_numeric_dtype(all_data[x_axis])):
        raise ValueError(
            f"Sequence Index '{x_axis}' must contain numerical or datetime values only")
    if not (is_datetime(all_data[y_axis]) or
            pd.api.types.is_numeric_dtype(all_data[y_axis])):
        raise ValueError(
            f"Column Name '{y_axis}' must contain numerical or datetime values only")

    fig = px.line(all_data, x=x_axis, y=y_axis, color=marker,
                  color_discrete_map={
                      'Real': PlotConfig.DATACEBO_DARK,
                      'Synthetic': PlotConfig.DATACEBO_GREEN
                  })
    if annotations:
        fig.add_annotation(annotations)

    if x_axis == 'sequence_index':
        fig.update_xaxes(title_text='Sequence Position')

    fig.update_layout(
        title_text=f"Real vs Synthetic Data for column: '{y_axis}'",
        plot_bgcolor=PlotConfig.BACKGROUND_COLOR,
        font={'size': PlotConfig.FONT_SIZE},
    )

    # Add min-max shading
    if 'min' in all_data and 'max' in all_data:
        fig.add_trace(
            go.Scatter(
                name='Real-Min',
                x=real_data[x_axis],
                y=real_data['min'],
                hoverinfo='skip',
                marker={'color': PlotConfig.DATACEBO_DARK_TRANSPARENT},
                showlegend=False,
                mode='lines'
            )
        )
        fig.add_trace(
            go.Scatter(
                name='Real-Max',
                x=real_data[x_axis],
                y=real_data['max'],
                hoverinfo='skip',
                marker={'color': PlotConfig.DATACEBO_DARK_TRANSPARENT},
                showlegend=False,
                mode='lines',
                fill='tonexty',
                fillcolor=PlotConfig.DATACEBO_DARK_TRANSPARENT,
            )
        )
        fig.add_trace(
            go.Scatter(
                name='Synthetic-Min',
                x=synthetic_data[x_axis],
                y=synthetic_data['min'],
                hoverinfo='skip',
                marker={'color': PlotConfig.DATACEBO_GREEN_TRANSPARENT},
                showlegend=False,
                mode='lines'
            )
        )
        fig.add_trace(
            go.Scatter(
                name='Synthetic-Max',
                x=synthetic_data[x_axis],
                y=synthetic_data['max'],
                hoverinfo='skip',
                marker={'color': PlotConfig.DATACEBO_GREEN_TRANSPARENT},
                showlegend=False,
                mode='lines',
                fill='tonexty',
                fillcolor=PlotConfig.DATACEBO_GREEN_TRANSPARENT,
            )
        )
    return fig


def get_column_line_plot(real_data, synthetic_data, column_name, metadata):
    """Return a line plot of the real and synthetic data.

    Args:
        real_data (pandas.DataFrame):
            The real table data.
        synthetic_column (pandas.Dataframe):
            The synthetic table data.
        column_name (str):
            The column name to be used as the y-axis of the graph
        metadata (dict):
            TimeSeries metadata dict. If not passed, the graph will
            use raw indices to build the graph and only separate the sequences
            into real and synthetic plots

    Returns:
        plotly.graph_objects._figure.Figure
    """
    real_column = real_data[column_name]
    synthetic_column = synthetic_data[column_name]

    missing_data_real = get_missing_percentage(real_column)
    missing_data_synthetic = get_missing_percentage(synthetic_column)
    show_missing_values = missing_data_real > 0 or missing_data_synthetic > 0

    annotations = None if not show_missing_values else {
        'xref': 'paper',
        'yref': 'paper',
        'x': 1.0,
        'y': 1.05,
        'showarrow': False,
        'text': (
                f'*Missing Values: Real Data ({missing_data_real}%), '
                f'Synthetic Data ({missing_data_synthetic}%)'
        ),
    }

    # Merge the real and synthetic data and add a flag ``Data`` to indicate each one.
    r_data = real_data.copy()
    s_data = synthetic_data.copy()

    # Check for sequence index to determine the x-axis values
    x_axis = 'sequence_index'
    y_axis = column_name
    if 'sequence_index' in metadata:
        x_axis = metadata['sequence_index']
        if 'sequence_key' in metadata:
            r_data = r_data.groupby(x_axis, as_index=False).agg(
                {
                    x_axis: 'first',
                    column_name: ['mean', 'min', 'max']
                }
            ).rename(columns={'mean': column_name, 'first': x_axis})
            s_data = s_data.groupby(x_axis, as_index=False).agg(
                {
                    x_axis: 'first',
                    column_name: ['mean', 'min', 'max']
                }
            ).rename(columns={'mean': column_name, 'first': x_axis})

            r_data.columns = r_data.columns.droplevel(0)
            s_data.columns = s_data.columns.droplevel(0)
    else:
        r_data['sequence_index'] = r_data.index
        s_data['sequence_index'] = s_data.index

    marker_name = 'Data'
    r_data[marker_name] = 'Real'
    s_data[marker_name] = 'Synthetic'

    # Generate plot
    fig = _generate_line_plot(
        real_data=r_data,
        synthetic_data=s_data,
        x_axis=x_axis,
        y_axis=y_axis,
        marker=marker_name,
        annotations=annotations
    )
    return fig
