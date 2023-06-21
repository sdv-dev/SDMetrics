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

DATACEBO_DARK = '#000036'
DATACEBO_LIGHT = '#01E0C9'
BACKGROUND_COLOR = '#F5F5F8'
CONTINUOUS_SDTYPES = ['numerical', 'datetime']
DISCRETE_SDTYPES = ['categorical', 'boolean']
DIAGNOSTIC_REPORT_RESULT_DETAILS = {
    'BoundaryAdherence': {
        'SUCCESS': (
            'The synthetic data follows over 90% of the min/max boundaries set by the real data'
        ),
        'WARNING': (
            'More than 10% the synthetic data does not follow the min/max boundaries set by '
            'the real data'
        ),
        'DANGER': (
            'More than 50% the synthetic data does not follow the min/max boundaries set by '
            'the real data'
        ),
    },
    'CategoryCoverage': {
        'SUCCESS': 'The synthetic data covers over 90% of the categories present in the real data',
        'WARNING': (
            'The synthetic data is missing more than 10% of the categories present in the '
            'real data'
        ),
        'DANGER': (
            'The synthetic data is missing more than 50% of the categories present in the '
            'real data'
        ),
    },
    'NewRowSynthesis': {
        'SUCCESS': 'Over 90% of the synthetic rows are not copies of the real data',
        'WARNING': 'More than 10% of the synthetic rows are copies of the real data',
        'DANGER': 'More than 50% of the synthetic rows are copies of the real data',
    },
    'RangeCoverage': {
        'SUCCESS': (
            'The synthetic data covers over 90% of the numerical ranges present in the real data'
        ),
        'WARNING': (
            'The synthetic data is missing more than 10% of the numerical ranges present in '
            'the real data'
        ),
        'DANGER': (
            'The synthetic data is missing more than 50% of the numerical ranges present in '
            'the real data'
        ),
    }
}


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


def print_results_for_level(out, results, level):
    """Print the result for a given level.

    Args:
        out:
            Where to write to.
        results (dict):
            The results.
        level (string):
            The level to print results for.
    """
    level_marks = {'SUCCESS': '✓', 'WARNING': '!', 'DANGER': 'x'}

    if len(results[level]) > 0:
        out.write(f'\n{level}:\n')
        for result in results[level]:
            out.write(f'{level_marks[level]} {result}\n')


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


def validate_multi_table_inputs(real_data, synthetic_data, metadata):
    """Validate multi-table inputs for report generation.

    Args:
        real_data (dict[str, DataFrame]):
            The real data.
        synthetic_data (dict[str, DataFrame]):
            The synthetic data.
        metadata (dict):
            The metadata, which contains each column's data type as well as relationships.
    """
    if not isinstance(metadata, dict):
        metadata = metadata.to_dict()

    for table in metadata['tables']:
        table_metadata = metadata['tables'][table]
        _validate_categorical_values(real_data[table],
                                     synthetic_data[table],
                                     table_metadata,
                                     table=table)

    for rel in metadata.get('relationships', []):
        parent_dtype = real_data[rel['parent_table_name']][rel['parent_primary_key']].dtype
        child_dtype = real_data[rel['child_table_name']][rel['child_foreign_key']].dtype
        if (parent_dtype == 'object' and child_dtype != 'object') or (
                parent_dtype != 'object' and child_dtype == 'object'):
            parent = rel['parent_table_name']
            parent_key = rel['parent_primary_key']
            child = rel['child_table_name']
            child_key = rel['child_foreign_key']
            error_msg = (f"The '{parent}' table and '{child}' table cannot be merged. Please "
                         f"make sure the primary key in '{parent}' ('{parent_key}') and the "
                         f"foreign key in '{child}' ('{child_key}') have the same data type.")
            raise ValueError(error_msg)


def validate_single_table_inputs(real_data, synthetic_data, metadata):
    """Validate single table inputs for report generation.

    Args:
        real_data (pandas.DataFrame):
            The real data.
        synthetic_data (pandas.DataFrame):
            The synthetic data.
        metadata (dict):
            The metadata, which contains each column's data type as well as relationships.
    """
    if not isinstance(metadata, dict):
        metadata = metadata.to_dict()

    _validate_categorical_values(real_data, synthetic_data, metadata)
