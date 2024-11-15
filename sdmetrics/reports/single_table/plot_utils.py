"""Utility methods for plotting metric scores."""

import numpy as np
import pandas as pd
import plotly.express as px
from plotly import graph_objects as go
from plotly.subplots import make_subplots

from sdmetrics.reports.utils import PlotConfig


def _get_column_shapes_data(score_breakdowns):
    """Convert the score breakdowns into the desired column shapes data format.

    Args:
        score_breakdowns (dict):
            A mapping of column shape metrics to the score breakdowns.

    Returns:
        pandas.DataFrame
    """
    column_names = []
    metrics = []
    scores = []
    for metric, score_breakdown in score_breakdowns.items():
        for column, result in score_breakdown.items():
            if 'score' not in result:
                continue

            if not np.isnan(result['score']):
                column_names.append(column)
                metrics.append(metric)
                scores.append(result['score'])

    return pd.DataFrame({'Column Name': column_names, 'Metric': metrics, 'Score': scores})


def get_column_shapes_plot(score_breakdowns, average_score=None):
    """Create a plot to show the column shape similarities.

    Args:
        score_breakdowns (dict):
            The score breakdowns of the column shape metrics.
        average_score (float):
            The average score. If None, the average score will be computed from
            ``score_breakdowns``.

    Returns:
        plotly.graph_objects._figure.Figure
    """
    data = _get_column_shapes_data(score_breakdowns)
    if average_score is None:
        average_score = data['Score'].mean()

    fig = px.bar(
        data,
        x='Column Name',
        y='Score',
        title=f'Data Quality: Column Shapes (Average Score={round(average_score, 2)})',
        category_orders={'group': data['Column Name']},
        color='Metric',
        color_discrete_map={
            'KSComplement': PlotConfig.DATACEBO_DARK,
            'TVComplement': PlotConfig.DATACEBO_BLUE,
        },
        pattern_shape='Metric',
        pattern_shape_sequence=['', '/'],
        hover_name='Column Name',
        hover_data={
            'Column Name': False,
            'Metric': True,
            'Score': True,
        },
    )

    fig.update_yaxes(range=[0, 1])

    fig.update_layout(
        xaxis_categoryorder='total ascending',
        plot_bgcolor=PlotConfig.BACKGROUND_COLOR,
        margin={'t': 150},
        font={'size': PlotConfig.FONT_SIZE},
    )

    return fig


def get_column_coverage_plot(score_breakdowns, average_score=None):
    """Create a plot to show the column coverage scores.

    Args:
        score_breakdowns (dict):
            The score breakdowns of the column shape metrics.
        average_score (float):
            The average score. If None, the average score will be computed from
            ``score_breakdowns``.

    Returns:
        plotly.graph_objects._figure.Figure
    """
    data = _get_column_shapes_data(score_breakdowns)
    data = data.rename(columns={'Score': 'Diagnostic Score'})
    if average_score is None:
        average_score = data['Diagnostic Score'].mean()

    fig = px.bar(
        data,
        x='Column Name',
        y='Diagnostic Score',
        title=f'Data Diagnostics: Column Coverage (Average Score={round(average_score, 2)})',
        category_orders={'group': data['Column Name']},
        color='Metric',
        color_discrete_map={
            'RangeCoverage': PlotConfig.DATACEBO_DARK,
            'CategoryCoverage': PlotConfig.DATACEBO_BLUE,
        },
        pattern_shape='Metric',
        pattern_shape_sequence=['', '/'],
        hover_name='Column Name',
        hover_data={
            'Column Name': False,
            'Metric': True,
            'Diagnostic Score': True,
        },
    )

    fig.update_yaxes(range=[0, 1])

    fig.update_layout(
        xaxis_categoryorder='total ascending',
        plot_bgcolor=PlotConfig.BACKGROUND_COLOR,
        margin={'t': 150},
        font={'size': PlotConfig.FONT_SIZE},
    )

    return fig


def get_column_boundaries_plot(score_breakdowns, average_score=None):
    """Create a plot to show the column boundary scores.

    Args:
        score_breakdowns (dict):
            The score breakdowns of the column shape metrics.
        average_score (float):
            The average score. If None, the average score will be computed from
            ``score_breakdowns``.

    Returns:
        plotly.graph_objects._figure.Figure
    """
    data = _get_column_shapes_data(score_breakdowns)
    data = data.rename(columns={'Score': 'Diagnostic Score'})
    if average_score is None:
        average_score = data['Diagnostic Score'].mean()

    fig = px.bar(
        data,
        x='Column Name',
        y='Diagnostic Score',
        title=f'Data Diagnostics: Column Boundaries (Average Score={round(average_score, 2)})',
        category_orders={'group': data['Column Name']},
        color='Metric',
        color_discrete_map={
            'BoundaryAdherence': PlotConfig.DATACEBO_DARK,
        },
        pattern_shape='Metric',
        pattern_shape_sequence=['', '/'],
        hover_name='Column Name',
        hover_data={
            'Column Name': False,
            'Metric': True,
            'Diagnostic Score': True,
        },
    )

    fig.update_yaxes(range=[0, 1])

    fig.update_layout(
        xaxis_categoryorder='total ascending',
        plot_bgcolor=PlotConfig.BACKGROUND_COLOR,
        margin={'t': 150},
        font={'size': PlotConfig.FONT_SIZE},
    )

    return fig


def _get_similarity_correlation_matrix(score_breakdowns, columns):
    """Convert the column pair score breakdowns to a similiarity correlation matrix.

    Args:
        score_breakdowns (dict):
            Mapping of metric to the score breakdown result.
        columns (list[string] or set[string]):
            A list or set of column names.

    Returns:
        pandas.DataFrame
    """
    if isinstance(columns, set):
        columns = list(columns)

    similarity_correlation = pd.DataFrame(
        index=columns,
        columns=columns,
        dtype='float',
    )
    np.fill_diagonal(similarity_correlation.to_numpy(), 1.0)

    for metric, score_breakdown in score_breakdowns.items():
        for column_pair, result in score_breakdown.items():
            column1, column2 = column_pair
            similarity_correlation.loc[column1, column2] = result['score']
            similarity_correlation.loc[column2, column1] = result['score']

    return similarity_correlation


def _get_numerical_correlation_matrices(score_breakdowns):
    """Convert the column pair score breakdowns to a numerical correlation matrix.

    Args:
        score_breakdowns (dict):
            Mapping of metric to the score breakdown result.

    Returns:
        (pandas.DataFrame, pandas.DataFrame):
            The real and synthetic numerical correlation matrices.
    """
    columns = []
    valid_score_breakdowns = {
        key: breakdown
        for key, breakdown in score_breakdowns['CorrelationSimilarity'].items()
        if not np.isnan(breakdown['score'])
    }
    for cols, _ in valid_score_breakdowns.items():
        if cols[0] not in columns:
            columns.append(cols[0])
        if cols[1] not in columns:
            columns.append(cols[1])

    real_correlation = pd.DataFrame(
        index=columns,
        columns=columns,
        dtype='float',
    )
    synthetic_correlation = pd.DataFrame(
        index=columns,
        columns=columns,
        dtype='float',
    )
    np.fill_diagonal(real_correlation.to_numpy(), 1.0)
    np.fill_diagonal(synthetic_correlation.to_numpy(), 1.0)

    for column_pair, result in valid_score_breakdowns.items():
        column1, column2 = column_pair
        real_correlation.loc[column1, column2] = result['real']
        real_correlation.loc[column2, column1] = result['real']
        synthetic_correlation.loc[column1, column2] = result['synthetic']
        synthetic_correlation.loc[column2, column1] = result['synthetic']

    return (real_correlation, synthetic_correlation)


def get_column_pairs_plot(score_breakdowns, average_score=None):
    """Create a plot to show the column pairs data.

    This plot will have one graph in the top row and two in the bottom row.

    Args:
        score_breakdowns (dict):
            The score breakdowns of the column pairs metric scores.
        average_score (float):
            The average score. If None, the average score will be computed from
            ``score_breakdowns``.

    Returns:
        plotly.graph_objects._figure.Figure
    """
    all_columns = []
    all_scores = []
    for _, score_breakdown in score_breakdowns.items():
        for column_pair, result in score_breakdown.items():
            all_columns.append(column_pair[0])
            all_columns.append(column_pair[1])
            all_scores.append(result['score'])

    if average_score is None:
        average_score = np.mean(all_scores)

    similarity_correlation = _get_similarity_correlation_matrix(score_breakdowns, set(all_columns))
    real_correlation, synthetic_correlation = _get_numerical_correlation_matrices(score_breakdowns)

    fig = make_subplots(
        rows=2,
        cols=2,
        subplot_titles=[
            'Real vs. Synthetic Similarity',
            'Numerical Correlation (Real Data)',
            'Numerical Correlation (Synthetic Data)',
        ],
        specs=[[{'colspan': 2, 'l': 0.26, 'r': 0.26}, None], [{}, {}]],
    )

    # Top row: Overall Similarity Graph
    fig.add_trace(
        go.Heatmap(
            x=similarity_correlation.columns,
            y=similarity_correlation.columns,
            z=similarity_correlation.round(2),
            coloraxis='coloraxis',
            xaxis='x',
            yaxis='y',
            hovertemplate=(
                '<b>Column Pair</b><br>(%{x},%{y})<br><br>Similarity: %{z}<extra></extra>'
            ),
        ),
        1,
        1,
    )

    # Real correlation heatmap
    fig.add_trace(
        go.Heatmap(
            x=real_correlation.columns,
            y=real_correlation.columns,
            z=real_correlation.round(2),
            coloraxis='coloraxis2',
            xaxis='x2',
            yaxis='y2',
            # Compare against synthetic data in the tooltip.
            customdata=synthetic_correlation.round(2),
            hovertemplate=(
                '<b>Correlation</b><br>(%{x},%{y})<br><br>Real: %{z}'
                '<br>(vs. Synthetic: %{customdata})<extra></extra>'
            ),
        ),
        2,
        1,
    )

    # Synthetic correlation heatmap
    fig.add_trace(
        go.Heatmap(
            x=synthetic_correlation.columns,
            y=synthetic_correlation.columns,
            z=synthetic_correlation.round(2),
            coloraxis='coloraxis2',
            xaxis='x3',
            yaxis='y3',
            # Compare against real data in the tooltip.
            customdata=real_correlation.round(2),
            hovertemplate=(
                '<b>Correlation</b><br>(%{x},%{y})<br><br>Synthetic: '
                '%{z}<br>(vs. Real: %{customdata})<extra></extra>'
            ),
        ),
        2,
        2,
    )

    fig.update_layout(
        title_text=f'Data Quality: Column Pair Trends (Average Score={round(average_score, 2)})',
        # Similarity heatmap color axis
        coloraxis={
            'colorbar_len': 0.5,
            'colorbar_x': 0.8,
            'colorbar_y': 0.8,
            'cmin': 0,
            'cmax': 1,
            'colorscale': [PlotConfig.RED, PlotConfig.ORANGE, PlotConfig.GREEN],
        },
        # Correlation heatmaps color axis
        coloraxis2={
            'colorbar_len': 0.5,
            'colorbar_y': 0.2,
            'cmin': -1,
            'cmax': 1,
            'colorscale': [
                PlotConfig.DATACEBO_BLUE,
                PlotConfig.DATACEBO_DARK,
                PlotConfig.DATACEBO_GREEN,
            ],
        },
        # Sync the zoom and pan of the bottom 2 graphs
        yaxis3={'visible': False, 'matches': 'y2'},
        xaxis3={'matches': 'x2'},
        height=900,
        width=900,
        font={'size': PlotConfig.FONT_SIZE},
    )

    fig.update_yaxes(autorange='reversed')

    return fig


def get_synthesis_plot(score_breakdown):
    """Create a plot to show the synthesis score.

    Args:
        score_breakdown (dict):
            The new row synthesis score breakdown.

    Returns:
        plotly.graph_objects._figure.Figure
    """
    labels = ['Exact Matches', 'Novel Rows']
    values = [
        score_breakdown.get('num_matched_rows', np.nan),
        score_breakdown.get('num_new_rows', np.nan),
    ]

    average_score = score_breakdown.get('score', np.nan)
    if not np.isnan(average_score):
        average_score = round(average_score, 2)

    fig = px.pie(
        values=values,
        names=labels,
        color=['Exact Matches', 'Novel Rows'],
        color_discrete_map={'Exact Matches': PlotConfig.ORANGE, 'Novel Rows': PlotConfig.GREEN},
        hole=0.4,
        title=f'Data Diagnostic: Synthesis (Score={average_score})',
    )

    fig.update_traces(hovertemplate='<b>%{label}</b><br>%{value} rows')
    fig.update_layout(font={'size': PlotConfig.FONT_SIZE})

    return fig
