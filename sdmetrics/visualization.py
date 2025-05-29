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
        real_column (pandas.Series or None):
            The real data for the desired column. If None this data will not be graphed.
        synthetic_column (pandas.Series or None):
            The synthetic data for the desired column. If None this data will not be graphed.
        plot_kwargs (dict, optional):
            Dictionary of keyword arguments to pass to px.histogram. Keyword arguments
            provided this way will overwrite defaults.

    Returns:
        plotly.graph_objects._figure.Figure
    """
    all_data = pd.DataFrame()
    color_sequence = []
    if real_data is not None:
        all_data = pd.concat([all_data, real_data], axis=0, ignore_index=True)
        color_sequence.append(PlotConfig.DATACEBO_DARK)
    if synthetic_data is not None:
        all_data = pd.concat([all_data, synthetic_data], axis=0, ignore_index=True)
        color_sequence.append(PlotConfig.DATACEBO_GREEN)

    histogram_kwargs = {
        'x': 'values',
        'color': 'Data',
        'barmode': 'group',
        'color_discrete_sequence': color_sequence,
        'pattern_shape': 'Data',
        'pattern_shape_sequence': ['', '/'],
        'histnorm': 'probability density',
    }
    histogram_kwargs.update(plot_kwargs)
    fig = px.histogram(all_data, **histogram_kwargs)

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
    unique_values = all_data['Data'].unique()

    if len(columns) != 2:
        raise ValueError('Generating a heatmap plot requires exactly two columns for the axis.')

    fig = px.density_heatmap(
        all_data, x=columns[0], y=columns[1], facet_col='Data', histnorm='probability'
    )

    title = ' vs. '.join(unique_values)
    title += f" Data for columns '{columns[0]}' and '{columns[1]}'"

    fig.update_layout(
        title_text=title,
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
            'Synthetic': PlotConfig.DATACEBO_GREEN,
        },
    )

    unique_values = all_data['Data'].unique()
    title = ' vs. '.join(unique_values)
    title += f" Data for columns '{columns[0]}' and '{columns[1]}'"
    fig.update_layout(
        title=title,
        plot_bgcolor=PlotConfig.BACKGROUND_COLOR,
        font={'size': PlotConfig.FONT_SIZE},
    )

    return fig


def _generate_violin_plot(data, columns):
    """Return a violin plot for a given column pair."""
    fig = px.violin(
        data,
        x=columns[0],
        y=columns[1],
        box=False,
        violinmode='overlay',
        color='Data',
        color_discrete_map={
            'Real': PlotConfig.DATACEBO_DARK,
            'Synthetic': PlotConfig.DATACEBO_GREEN,
        },
    )

    unique_values = data['Data'].unique()
    title = ' vs. '.join(unique_values)
    title += f" Data for columns '{columns[0]}' and '{columns[1]}'"
    fig.update_layout(
        title=title,
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
    if len(columns) != 2:
        raise ValueError('Generating a scatter plot requires exactly two columns for the axis.')

    unique_values = all_data['Data'].unique()
    fig = px.scatter(
        all_data,
        x=columns[0],
        y=columns[1],
        color='Data',
        color_discrete_map={
            'Real': PlotConfig.DATACEBO_DARK,
            'Synthetic': PlotConfig.DATACEBO_GREEN,
        },
        symbol='Data',
    )

    title = ' vs. '.join(unique_values)
    title += f" Data for columns '{columns[0]}' and '{columns[1]}'"

    fig.update_layout(
        title=title,
        plot_bgcolor=PlotConfig.BACKGROUND_COLOR,
        font={'size': PlotConfig.FONT_SIZE},
    )

    return fig


def _generate_column_distplot(real_data, synthetic_data, plot_kwargs={}):
    """Plot the real and synthetic data as a distplot.

    Args:
        real_data (pandas.DataFrame or None):
            The real data for the desired column. If None this data will not be graphed.
        synthetic_data (pandas.DataFrame or None):
            The synthetic data for the desired column. If None this data will not be graphed.
        plot_kwargs (dict, optional):
            Dictionary of keyword arguments to pass to px.histogram. Keyword arguments
            provided this way will overwrite defaults.

    Returns:
        plotly.graph_objects._figure.Figure
    """
    hist_data = []
    col_names = []
    colors = []
    if real_data is not None:
        hist_data.append(real_data['values'])
        col_names.append('Real')
        colors.append(PlotConfig.DATACEBO_DARK)
    if synthetic_data is not None:
        hist_data.append(synthetic_data['values'])
        col_names.append('Synthetic')
        colors.append(PlotConfig.DATACEBO_GREEN)

    default_distplot_kwargs = {
        'show_hist': False,
        'show_rug': False,
        'colors': colors,
    }

    has_data = any(len(data) > 0 for data in hist_data)

    if has_data:
        return ff.create_distplot(
            hist_data,
            col_names,
            **{**default_distplot_kwargs, **plot_kwargs},
        )

    return go.Figure()


def _generate_column_plot(
    real_column, synthetic_column, plot_type, plot_kwargs={}, plot_title=None, x_label=None
):
    """Generate a plot of the real and synthetic data.

    Args:
        real_column (pandas.Series or None):
            The real data for the desired column. If None this data will not be graphed.
        synthetic_column (pandas.Series or None)
            The synthetic data for the desired column. If None this data will not be graphed.
        plot_type (str):
            The type of plot to use. Must be one of 'bar' or 'distplot'.
        hist_kwargs (dict, optional):
            Dictionary of keyword arguments to pass to px.histogram. Keyword arguments
            provided this way will overwrite defaults.
        plot_title (str, optional):
            Title to use for the plot. Defaults to 'Real vs. Synthetic Data for column {column}'
        x_label (str, optional):
            Label to use for x-axis. Defaults to 'Value'.

    Returns:
        plotly.graph_objects._figure.Figure
    """
    if real_column is None and synthetic_column is None:
        raise ValueError('No data provided to plot. Please provide either real or synthetic data.')

    if plot_type not in ['bar', 'distplot']:
        raise ValueError(
            f"Unrecognized plot_type '{plot_type}'. Please use one of 'bar' or 'distplot'"
        )

    column_name = ''
    missing_data_real = 0
    missing_data_synthetic = 0
    col_dtype = None
    col_names = []
    title = ''
    if real_column is not None and hasattr(real_column, 'name'):
        column_name = real_column.name
    elif synthetic_column is not None and hasattr(synthetic_column, 'name'):
        column_name = synthetic_column.name

    real_data = None
    if real_column is not None:
        missing_data_real = get_missing_percentage(real_column)
        real_data = pd.DataFrame({'values': real_column.copy().dropna()})
        real_data['Data'] = 'Real'
        col_dtype = real_column.dtype
        col_names.append('Real')
        title += 'Real vs. '

    synthetic_data = None
    if synthetic_column is not None:
        missing_data_synthetic = get_missing_percentage(synthetic_column)
        synthetic_data = pd.DataFrame({'values': synthetic_column.copy().dropna()})
        synthetic_data['Data'] = 'Synthetic'
        col_names.append('Synthetic')
        title += 'Synthetic vs. '
        if col_dtype is None:
            col_dtype = synthetic_column.dtype

    title = title[:-4]
    title += f"Data for column '{column_name}'"

    is_datetime_sdtype = False
    if is_datetime64_dtype(col_dtype):
        is_datetime_sdtype = True
        if real_data is not None:
            real_data['values'] = real_data['values'].astype('int64')
        if synthetic_data is not None:
            synthetic_data['values'] = synthetic_data['values'].astype('int64')

    trace_args = {}

    if plot_type == 'bar':
        fig = _generate_column_bar_plot(real_data, synthetic_data, plot_kwargs)
    elif plot_type == 'distplot':
        x_label = x_label or 'Value'
        fig = _generate_column_distplot(real_data, synthetic_data, plot_kwargs)
        trace_args = {'fill': 'tozeroy'}

    annotations = []
    if fig.data:
        for idx, name in enumerate(col_names):
            fig.update_traces(
                x=pd.to_datetime(fig.data[idx].x) if is_datetime_sdtype else fig.data[idx].x,
                hovertemplate=f'<b>{name}</b><br>Frequency: %{{y}}<extra></extra>',
                selector={'name': name},
                **trace_args,
            )
    else:
        annotations.append({
            'xref': 'paper',
            'yref': 'paper',
            'x': 0.5,
            'y': 0.5,
            'showarrow': False,
            'text': 'No data to visualize',
            'font': {'size': PlotConfig.FONT_SIZE * 2},
        })

    show_missing_values = missing_data_real > 0 or missing_data_synthetic > 0
    text = '*Missing Values:'
    if real_column is not None and show_missing_values:
        text += f' Real Data ({missing_data_real}%), '
    if synthetic_column is not None and show_missing_values:
        text += f'Synthetic Data ({missing_data_synthetic}%), '

    text = text[:-2]

    if show_missing_values:
        annotations.append({
            'xref': 'paper',
            'yref': 'paper',
            'x': 1.0,
            'y': 1.05,
            'showarrow': False,
            'text': text,
        })

    if not plot_title:
        plot_title = title

    if not x_label:
        x_label = 'Value'

    fig.update_layout(
        title=plot_title,
        xaxis_title=x_label,
        yaxis_title='Frequency',
        plot_bgcolor=PlotConfig.BACKGROUND_COLOR,
        annotations=annotations,
        font={'size': PlotConfig.FONT_SIZE},
    )
    return fig


def _get_max_between_datasets(real_data, synthetic_data):
    if synthetic_data is None and real_data is None:
        raise ValueError('Cannot get max between two None values.')
    if real_data is None:
        return max(synthetic_data)
    elif synthetic_data is None:
        return max(real_data)
    return max(max(real_data), max(synthetic_data))


def _get_min_between_datasets(real_data, synthetic_data):
    if synthetic_data is None and real_data is None:
        raise ValueError('Cannot get min between two None values.')
    if real_data is None:
        return min(synthetic_data)
    elif synthetic_data is None:
        return min(real_data)
    return min(min(real_data), min(synthetic_data))


def _generate_cardinality_plot(
    real_data, synthetic_data, parent_primary_key, child_foreign_key, plot_type='bar'
):
    plot_title = (
        f"Relationship (child foreign key='{child_foreign_key}' and parent "
        f"primary key='{parent_primary_key}')"
    )
    x_label = '# of Children (per Parent)'

    plot_kwargs = {}
    if plot_type == 'bar':
        max_cardinality = _get_max_between_datasets(real_data, synthetic_data)
        min_cardinality = _get_min_between_datasets(real_data, synthetic_data)
        plot_kwargs = {'nbins': max_cardinality - min_cardinality + 1}

    return _generate_column_plot(
        real_data, synthetic_data, plot_type, plot_kwargs, plot_title, x_label
    )


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
def get_cardinality_plot(
    real_data,
    synthetic_data,
    child_table_name,
    parent_table_name,
    child_foreign_key,
    parent_primary_key,
    plot_type='bar',
):
    """Return a plot of the cardinality of the parent-child relationship.

    Args:
        real_data (dict or None):
            The real data. If None this data will not be graphed.
        synthetic_data (dict or None):
            The synthetic data. If None this data will not be graphed.
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
        raise ValueError(f"Invalid plot_type '{plot_type}'. Please use one of ['bar', 'distplot'].")

    if real_data is None and synthetic_data is None:
        raise ValueError('No data provided to plot. Please provide either real or synthetic data.')

    real_cardinality = None
    synth_cardinality = None

    if real_data is not None:
        real_cardinality = _get_cardinality(
            real_data[parent_table_name],
            real_data[child_table_name],
            parent_primary_key,
            child_foreign_key,
        )

    if synthetic_data is not None:
        synth_cardinality = _get_cardinality(
            synthetic_data[parent_table_name],
            synthetic_data[child_table_name],
            parent_primary_key,
            child_foreign_key,
        )

    fig = _generate_cardinality_plot(
        real_cardinality,
        synth_cardinality,
        parent_primary_key,
        child_foreign_key,
        plot_type=plot_type,
    )

    return fig


@set_plotly_config
def get_column_plot(real_data, synthetic_data, column_name, plot_type=None):
    """Return a plot of the real and synthetic data for a given column.

    Args:
        real_data (pandas.DataFrame or None):
            The real table data. If None this data will not be graphed.
        synthetic_data (pandas.DataFrame or None):
            The synthetic table data. If None this data will not be graphed.
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
    if real_data is None and synthetic_data is None:
        raise ValueError('No data provided to plot. Please provide either real or synthetic data.')

    if plot_type not in ['bar', 'distplot', None]:
        raise ValueError(
            f"Invalid plot_type '{plot_type}'. Please use one of ['bar', 'distplot', None]."
        )

    column = None
    real_column = None
    synthetic_column = None
    if real_data is not None:
        if column_name not in real_data.columns:
            raise ValueError(f"Column '{column_name}' not found in real table data.")

        column = real_data[column_name]
        real_column = real_data[column_name]

    if synthetic_data is not None:
        if column_name not in synthetic_data.columns:
            raise ValueError(f"Column '{column_name}' not found in synthetic table data.")
        if column is None:
            column = synthetic_data[column_name]

        synthetic_column = synthetic_data[column_name]

    real_constant = real_column is not None and real_column.nunique() == 1
    synthetic_constant = synthetic_column is not None and synthetic_column.nunique() == 1
    column_is_constant = real_constant or synthetic_constant
    if plot_type is None:
        column_is_datetime = is_datetime(column)
        dtype = column.dropna().infer_objects().dtype.kind
        if column_is_datetime or dtype in ('i', 'f') and not column_is_constant:
            plot_type = 'distplot'
        else:
            plot_type = 'bar'
    elif plot_type == 'distplot' and column_is_constant:
        raise ValueError(
            f"Plot type 'distplot' cannot be created because column '{column_name}'"
            ' has a constant value inside the real or synthetic data. To render a'
            " visualization, please update the plot_type to 'bar'."
        )

    fig = _generate_column_plot(real_column, synthetic_column, plot_type)

    return fig


@set_plotly_config
def get_column_pair_plot(real_data, synthetic_data, column_names, plot_type=None):
    """Return a plot of the real and synthetic data for a given column pair.

    Args:
        real_data (pandas.DataFrame or None):
            The real table data. If None this data will not be graphed.
        synthetic_column (pandas.Dataframe or None):
            The synthetic table data. If None this data will not be graphed.
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

    if real_data is None and synthetic_data is None:
        raise ValueError('No data provided to plot. Please provide either real or synthetic data.')

    if real_data is not None:
        if not set(column_names).issubset(real_data.columns):
            raise ValueError(
                f'Missing column(s) {set(column_names) - set(real_data.columns)} in real data.'
            )
        real_data = real_data[column_names]

    if synthetic_data is not None:
        if not set(column_names).issubset(synthetic_data.columns):
            raise ValueError(
                f'Missing column(s) {set(column_names) - set(synthetic_data.columns)} '
                'in synthetic data.'
            )
        synthetic_data = synthetic_data[column_names]

    if plot_type not in ['box', 'heatmap', 'scatter', 'violin', None]:
        raise ValueError(
            f"Invalid plot_type '{plot_type}'. Please use one of "
            "['box', 'heatmap', 'scatter', 'violin', None]."
        )

    if plot_type is None:
        plot_type = []
        for column_name in column_names:
            if real_data is not None:
                column = real_data[column_name]
            else:
                column = synthetic_data[column_name]
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
    all_data = pd.DataFrame()
    if real_data is not None:
        real_data = real_data.copy()
        real_data['Data'] = 'Real'
        all_data = pd.concat([all_data, real_data], axis=0, ignore_index=True)
    if synthetic_data is not None:
        synthetic_data = synthetic_data.copy()
        synthetic_data['Data'] = 'Synthetic'
        all_data = pd.concat([all_data, synthetic_data], axis=0, ignore_index=True)

    if plot_type == 'scatter':
        return _generate_scatter_plot(all_data, column_names)
    elif plot_type == 'heatmap':
        return _generate_heatmap_plot(all_data, column_names)
    elif plot_type == 'violin':
        return _generate_violin_plot(all_data, column_names)

    return _generate_box_plot(all_data, column_names)


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
    if not (is_datetime(all_data[x_axis]) or pd.api.types.is_numeric_dtype(all_data[x_axis])):
        raise ValueError(
            f"Sequence Index '{x_axis}' must contain numerical or datetime values only"
        )
    if not (is_datetime(all_data[y_axis]) or pd.api.types.is_numeric_dtype(all_data[y_axis])):
        raise ValueError(f"Column Name '{y_axis}' must contain numerical or datetime values only")

    fig = px.line(
        all_data,
        x=x_axis,
        y=y_axis,
        color=marker,
        color_discrete_map={
            'Real': PlotConfig.DATACEBO_DARK,
            'Synthetic': PlotConfig.DATACEBO_GREEN,
        },
    )
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
                mode='lines',
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
                mode='lines',
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

    annotations = (
        None
        if not show_missing_values
        else {
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
    )

    # Merge the real and synthetic data and add a flag ``Data`` to indicate each one.
    r_data = real_data.copy()
    s_data = synthetic_data.copy()

    # Check for sequence index to determine the x-axis values
    x_axis = 'sequence_index'
    y_axis = column_name
    if 'sequence_index' in metadata:
        x_axis = metadata['sequence_index']
        if 'sequence_key' in metadata:
            r_data = (
                r_data.groupby(x_axis, as_index=False)
                .agg({x_axis: 'first', column_name: ['mean', 'min', 'max']})
                .rename(columns={'mean': column_name, 'first': x_axis})
            )
            s_data = (
                s_data.groupby(x_axis, as_index=False)
                .agg({x_axis: 'first', column_name: ['mean', 'min', 'max']})
                .rename(columns={'mean': column_name, 'first': x_axis})
            )

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
        annotations=annotations,
    )
    return fig
