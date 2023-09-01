"""Report utility methods."""

import copy
import itertools
import warnings

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.figure_factory as ff
from pandas.core.tools.datetimes import _guess_datetime_format_for_array

from sdmetrics.utils import (
    get_alternate_keys, get_columns_from_metadata, get_missing_percentage,
    get_type_from_column_meta, is_datetime)

CONTINUOUS_SDTYPES = ['numerical', 'datetime']
DISCRETE_SDTYPES = ['categorical', 'boolean']


class PlotConfig:
    """Custom plot settings for visualizations."""

    GREEN = '#36B37E'
    RED = '#FF0000'
    ORANGE = '#F16141'
    DATACEBO_DARK = '#000036'
    DATACEBO_GREEN = '#01E0C9'
    DATACEBO_BLUE = '#03AFF1'
    BACKGROUND_COLOR = '#F5F5F8'
    FONT_SIZE = 18


def convert_to_datetime(column_data, datetime_format=None):
    """Convert a column data to pandas datetime.

    Args:
        column_data (pandas.Series):
            The column data
        format (str):
            Optional string format of datetime. If ``None``, will attempt to infer the datetime
            format from the column data. Defaults to ``None``.

    Returns:
        pandas.Series:
            The converted column data.
    """
    if is_datetime(column_data):
        return column_data

    if datetime_format is None:
        datetime_format = _guess_datetime_format_for_array(column_data.astype(str).to_numpy())

    return pd.to_datetime(column_data, format=datetime_format)


def convert_datetime_columns(real_column, synthetic_column, col_metadata):
    """Convert a real and a synthetic column to pandas datetime.

    Args:
        real_data (pandas.Series):
            The real column data
        synthetic_column (pandas.Series):
            The synthetic column data
        col_metadata:
            The metadata associated with the column

    Returns:
        (pandas.Series, pandas.Series):
            The converted real and synthetic column data.
    """
    datetime_format = col_metadata.get('format') or col_metadata.get('datetime_format')
    return (convert_to_datetime(real_column, datetime_format),
            convert_to_datetime(synthetic_column, datetime_format))


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

    missing_data_real = get_missing_percentage(real_column)
    missing_data_synthetic = get_missing_percentage(synthetic_column)

    all_data = pd.concat([real_data, synthetic_data], axis=0, ignore_index=True)

    fig = px.histogram(
        all_data,
        x='values',
        color='Data',
        barmode='group',
        color_discrete_sequence=[PlotConfig.DATACEBO_DARK, PlotConfig.DATACEBO_GREEN],
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
        plot_bgcolor=PlotConfig.BACKGROUND_COLOR,
        annotations=annotations,
        font={'size': PlotConfig.FONT_SIZE},
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
    missing_data_real = get_missing_percentage(real_column)
    missing_data_synthetic = get_missing_percentage(synthetic_column)

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
        colors=[PlotConfig.DATACEBO_DARK, PlotConfig.DATACEBO_GREEN]
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
        plot_bgcolor=PlotConfig.BACKGROUND_COLOR,
        annotations=annotations,
        font={'size': PlotConfig.FONT_SIZE},
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
    columns = get_columns_from_metadata(metadata)
    if column_name not in columns:
        raise ValueError(f"Column '{column_name}' not found in metadata.")
    elif 'sdtype' not in columns[column_name]:
        raise ValueError(f"Metadata for column '{column_name}' missing 'type' information.")
    if column_name not in real_data.columns:
        raise ValueError(f"Column '{column_name}' not found in real table data.")
    if column_name not in synthetic_data.columns:
        raise ValueError(f"Column '{column_name}' not found in synthetic table data.")

    column_meta = columns[column_name]
    sdtype = get_type_from_column_meta(columns[column_name])
    if sdtype == 'datetime':
        real_column, synthetic_column = convert_datetime_columns(
            real_data[column_name],
            synthetic_data[column_name],
            column_meta
        )
    else:
        real_column = real_data[column_name]
        synthetic_column = synthetic_data[column_name]
    if sdtype in CONTINUOUS_SDTYPES:
        fig = make_continuous_column_plot(real_column, synthetic_column, sdtype)
    elif sdtype in DISCRETE_SDTYPES:
        fig = make_discrete_column_plot(real_column, synthetic_column, sdtype)
    else:
        raise ValueError(f"sdtype of type '{sdtype}' not recognized.")

    return fig


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
        coloraxis={'colorscale': [PlotConfig.DATACEBO_DARK, PlotConfig.DATACEBO_GREEN]},
        font={'size': PlotConfig.FONT_SIZE},
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


def get_column_pair_plot(real_data, synthetic_data, column_names, metadata):
    """Return a plot of the real and synthetic data for a given column pair.

    Args:
        real_data (pandas.DataFrame):
            The real table data.
        synthetic_column (pandas.Dataframe):
            The synthetic table data.
        column_names (list[string]):
            The names of the two columns to plot.
        metadata (dict):
            The table metadata.

    Returns:
        plotly.graph_objects._figure.Figure
    """
    all_columns = get_columns_from_metadata(metadata)
    invalid_columns = [column for column in column_names if column not in all_columns]
    if invalid_columns:
        raise ValueError(f"Column(s) `{'`, `'.join(invalid_columns)}` not found in metadata.")
    else:
        invalid_columns = [
            column for column in column_names if 'sdtype' not in all_columns[column]
        ]
        if invalid_columns:
            raise ValueError(f"Metadata for column(s) `{'`, `'.join(invalid_columns)}` "
                             "missing 'type' information.")

    invalid_columns = [column for column in column_names if column not in real_data.columns]
    if invalid_columns:
        raise ValueError(f"Column(s) `{'`, `'.join(invalid_columns)}` not found "
                         'in the real table data.')

    invalid_columns = [column for column in column_names if column not in synthetic_data.columns]
    if invalid_columns:
        raise ValueError(f"Column(s) `{'`, `'.join(invalid_columns)}` not found "
                         'in the synthetic table data.')

    col_meta = (all_columns[column_names[0]], all_columns[column_names[1]])
    sdtypes = (
        get_type_from_column_meta(col_meta[0]),
        get_type_from_column_meta(col_meta[1]),
    )
    real_data = real_data[column_names]
    synthetic_data = synthetic_data[column_names]

    all_sdtypes = CONTINUOUS_SDTYPES + DISCRETE_SDTYPES
    invalid_sdtypes = [sdtype for sdtype in sdtypes if sdtype not in all_sdtypes]
    if invalid_sdtypes:
        raise ValueError(f"sdtype(s) of type `{'`, `'.join(invalid_sdtypes)}` not recognized.")

    if all([t in DISCRETE_SDTYPES for t in sdtypes]):
        return make_discrete_column_pair_plot(real_data, synthetic_data)

    for i, sdtype in enumerate(sdtypes):
        if sdtype == 'datetime':
            real_data[column_names[i]], synthetic_data[column_names[i]] = convert_datetime_columns(
                real_data[column_names[i]],
                synthetic_data[column_names[i]],
                col_meta[i]
            )

    if all([t in CONTINUOUS_SDTYPES for t in sdtypes]):
        return make_continuous_column_pair_plot(real_data, synthetic_data)
    else:
        return make_mixed_column_pair_plot(real_data, synthetic_data)


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
    child_counts = child_table[child_foreign_key].value_counts()
    child_per_parent = child_counts.reindex(parent_table[parent_primary_key], fill_value=0)
    child_per_parent = child_per_parent.reset_index()
    child_per_parent.columns = [parent_primary_key, '# children']

    cardinalities = child_per_parent.groupby('# children').size().reset_index(name='# parents')

    return cardinalities.sort_values('# children')


def _generate_cardinality_plot(data, parent_primary_key, child_foreign_key):
    """Generate a plot of the cardinality of the parent-child relationship.

    Args:
        data (pandas.DataFrame):
            The cardinality data.
        parent_primary_key (string):
            The name of the primary key column in the parent table.
        child_foreign_key (string):
            The name of the foreign key column in the child table.

    Returns:
        plotly.graph_objects._figure.Figure
    """
    fig = px.histogram(
        data,
        x='# children',
        y='# parents',
        color='data',
        barmode='group',
        color_discrete_sequence=[PlotConfig.DATACEBO_DARK, PlotConfig.DATACEBO_GREEN],
        pattern_shape='data',
        pattern_shape_sequence=['', '/'],
        nbins=max(data['# children']) - min(data['# children']) + 1,
        histnorm='probability density'
    )

    for name in ['Real', 'Synthetic']:
        fig.update_traces(
            hovertemplate=f'<b>{name}</b><br>Frequency: %{{y}}<extra></extra>',
            selector={'name': name}
        )

    title = (
        f"Relationship (child foreign key='{child_foreign_key}' and parent "
        f"primary key='{parent_primary_key}')"
    )
    fig.update_layout(
        title=title,
        xaxis_title='# of Children (per Parent)',
        yaxis_title='Frequency',
        plot_bgcolor=PlotConfig.BACKGROUND_COLOR,
        font={'size': PlotConfig.FONT_SIZE},
    )
    return fig


def get_cardinality_plot(real_data, synthetic_data, child_table_name, parent_table_name,
                         child_foreign_key, metadata):
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
        metadata (dict):
            The metadata.

    Returns:
        plotly.graph_objects._figure.Figure
    """
    parent_primary_key = None
    for relation in metadata.get('relationships', []):
        parent_match = relation['parent_table_name'] == parent_table_name
        child_match = relation['child_table_name'] == child_table_name
        foreign_key_match = relation['child_foreign_key'] == child_foreign_key
        if child_match and parent_match and foreign_key_match:
            parent_primary_key = relation['parent_primary_key']

    if parent_primary_key is None:
        raise ValueError(
            f"No relationship found between child table '{child_table_name}' and parent table "
            f"'{parent_table_name}' for the foreign key '{child_foreign_key}' "
            'in the metadata. Please update the metadata.'
        )

    real_cardinality = _get_cardinality(
        real_data[parent_table_name], real_data[child_table_name],
        parent_primary_key, child_foreign_key
    )
    synth_cardinality = _get_cardinality(
        synthetic_data[parent_table_name],
        synthetic_data[child_table_name],
        parent_primary_key, child_foreign_key
    )

    real_cardinality['data'] = 'Real'
    synth_cardinality['data'] = 'Synthetic'

    all_cardinality = pd.concat([real_cardinality, synth_cardinality], ignore_index=True)
    fig = _generate_cardinality_plot(
        all_cardinality, parent_primary_key, child_foreign_key
    )

    return fig


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

    for column_name, column_meta in get_columns_from_metadata(metadata).items():
        sdtype = get_type_from_column_meta(column_meta)

        if sdtype in ('numerical', 'datetime'):
            real_col = real_data[column_name]
            synthetic_col = synthetic_data[column_name]
            if sdtype == 'datetime':
                datetime_format = column_meta.get('format') or column_meta.get('datetime_format')
                if real_col.dtype == 'O' and datetime_format:
                    real_col = pd.to_datetime(real_col, format=datetime_format)
                    synthetic_col = pd.to_datetime(synthetic_col, format=datetime_format)

                real_col = pd.to_numeric(real_col)
                synthetic_col = pd.to_numeric(synthetic_col)

            bin_edges = np.histogram_bin_edges(real_col.dropna())
            binned_real_col = np.digitize(real_col, bins=bin_edges)
            binned_synthetic_col = np.digitize(synthetic_col, bins=bin_edges)

            binned_real[column_name] = binned_real_col
            binned_synthetic[column_name] = binned_synthetic_col
            get_columns_from_metadata(binned_metadata)[column_name] = {'sdtype': 'categorical'}

    return binned_real, binned_synthetic, binned_metadata


def _get_non_id_columns(metadata, binned_metadata):
    valid_sdtypes = ['numerical', 'categorical', 'boolean', 'datetime']
    alternate_keys = get_alternate_keys(metadata)
    non_id_columns = []
    for column, column_meta in get_columns_from_metadata(binned_metadata).items():
        is_key = column == metadata.get('primary_key', '') or column in alternate_keys
        if get_type_from_column_meta(column_meta) in valid_sdtypes and not is_key:
            non_id_columns.append(column)

    return non_id_columns


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
        keys_to_skip (list[tuple(str)] or None):
            A list of keys for which to skip computing the metric.

    Returns:
        dict:
            The metric results.
    """
    metric_results = {}

    binned_real, binned_synthetic, binned_metadata = discretize_table_data(
        real_data, synthetic_data, metadata)

    non_id_cols = _get_non_id_columns(metadata, binned_metadata)
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


def _validate_categorical_values(real_data, synthetic_data, metadata, table=None):
    """Get categorical values found in synthetic data but not real data for all columns.

    Args:
        real_data (pd.DataFrame):
            The real data.
        synthetic_data (pd.DataFrame):
            The synthetic data.
        metadata (dict):
            The metadata.
        table (str, optional):
            The name of the current table, if one exists
    """
    if table:
        warning_format = ('Unexpected values ({values}) in column "{column}" '
                          f'and table "{table}"')
    else:
        warning_format = 'Unexpected values ({values}) in column "{column}"'

    columns = get_columns_from_metadata(metadata)
    for column, column_meta in columns.items():
        column_type = get_type_from_column_meta(column_meta)
        if column_type == 'categorical':
            extra_categories = [
                value for value in synthetic_data[column].unique()
                if value not in real_data[column].unique()
            ]
            if extra_categories:
                value_list = '", "'.join(str(value) for value in extra_categories[:5])
                values = f'"{value_list}" + more' if len(
                    extra_categories) > 5 else f'"{value_list}"'
                warnings.warn(warning_format.format(values=values, column=column))
