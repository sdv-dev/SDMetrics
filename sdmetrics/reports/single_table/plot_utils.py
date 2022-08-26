"""Utility methods for plotting metric scores."""

import numpy as np
import pandas as pd
import plotly.express as px
from plotly import graph_objects as go
from plotly.subplots import make_subplots


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
            if not np.isnan(result['score']):
                column_names.append(column)
                metrics.append(metric)
                scores.append(result['score'])

    return pd.DataFrame({'Column Name': column_names, 'Metric': metrics, 'Quality Score': scores})


def get_column_shapes_plot(score_breakdowns):
    """Create a plot to show the column shape similarities.

    Args:
        score_breakdowns (dict):
            The score breakdowns of the column shape metrics.

    Returns:
        plotly.graph_objects._figure.Figure
    """
    data = _get_column_shapes_data(score_breakdowns)
    average_score = round(data['Quality Score'].mean(), 2)

    fig = px.bar(
        data,
        x='Column Name',
        y='Quality Score',
        title=f'Column Shapes Similarity (Average={average_score})',
        category_orders={'group': data['Column Name']},
        color='Metric',
        color_discrete_map={
            'KSComplement': '#000036',
            'TVComplement': '#03AFF1',
        },
        pattern_shape='Metric',
        pattern_shape_sequence=['', '/'],
        hover_name='Column Name',
        hover_data={
            'Column Name': False,
            'Metric': True,
            'Quality Score': True,
        },
    )

    fig.update_yaxes(range=[0, 1])

    fig.update_layout(
        xaxis_categoryorder='total ascending',
        plot_bgcolor='#F5F5F8'
    )

    return fig


def _get_similarity_correlation_matrix(score_breakdowns, real_correlation):
    """Convert the column pair score breakdowns to a similiarity correlation matrix.

    Args:
        score_breakdowns (dict):
            Mapping of metric to the score breakdown result.

    Returns:
        pandas.DataFrame
    """
    similarity_correlation = pd.DataFrame(
        index=real_correlation.index,
        columns=real_correlation.columns,
    )

    for metric, score_breakdown in score_breakdowns.items():
        for column_pair, result in score_breakdown.items():
            column1, column2 = column_pair
            similarity_correlation.loc[column1, column2] = result['score']
            similarity_correlation.loc[column2, column1] = result['score']

    return similarity_correlation


def get_column_pairs_plot(score_breakdowns, real_correlation, synthetic_correlation):
    """Create a plot to show the column pairs data.

    This plot will have one graph in the top row and two in the bottom row.

    Args:
        score_breakdowns (dict):
            The score breakdowns of the column pairs metric scores.
        real_correlation (pandas.DataFrame):
            Correlation matrix for the real data.
        synthetic_correlation (pandas.DataFrame):
            Correlation matrix for the synthetic data.

    Returns:
        plotly.graph_objects._figure.Figure
    """
    all_scores = [
        result['score'] for _, score_breakdown in score_breakdowns.items()
        for column_pair, result in score_breakdown.items()
    ]
    average_quality_score = round(np.mean(all_scores), 2)

    similarity_correlation = _get_similarity_correlation_matrix(score_breakdowns, real_correlation)

    fig = make_subplots(
        rows=2,
        cols=2,
        subplot_titles=[
            f'Column Pairs Similarity ({average_quality_score})',
            'Numerical Correlation (Real)',
            'Numerical Correlation (Synthetic)',
        ],
        specs=[[{'colspan': 2, 'l': 0.26, 'r': 0.26}, None], [{}, {}]])

    # Top row: Overall Similarity Graph
    fig.add_trace(
        go.Heatmap(
            x=similarity_correlation.columns,
            y=similarity_correlation.columns,
            z=similarity_correlation,
            coloraxis='coloraxis',
            xaxis='x',
            yaxis='y',
            hovertemplate=('<b>Column Pair</b><br>(%{x},%{y})<br><br>Similarity: '
                           '%{z:.2f}<extra></extra>'),
        ),
        1,
        1,
    )

    # Real correlation heatmap
    fig.add_trace(
        go.Heatmap(
            x=real_correlation.columns,
            y=real_correlation.columns,
            z=real_correlation,
            coloraxis='coloraxis2',
            xaxis='x2',
            yaxis='y2',
            # Compare against synthetic data in the tooltip.
            customdata=synthetic_correlation,
            hovertemplate=('<b>Correlation</b><br>(%{x},%{y})<br><br>Real: %{z:.2f}'
                           '<br>(vs. Synthetic: %{customdata:.2f})<extra></extra>'),
        ),
        2,
        1,
    )

    # Synthetic correlation heatmap
    fig.add_trace(
        go.Heatmap(
            x=synthetic_correlation.columns,
            y=synthetic_correlation.columns,
            z=synthetic_correlation,
            coloraxis='coloraxis2',
            xaxis='x3',
            yaxis='y3',
            # Compare against real data in the tooltip.
            customdata=real_correlation,
            hovertemplate=('<b>Correlation</b><br>(%{x},%{y})<br><br>Synthetic: '
                           '%{z:.2f}<br>(vs. Real: %{customdata:.2f})<extra></extra>'),
        ),
        2,
        2,
    )

    fig.update_layout(
        title_text='Column Pair Trends',
        # Similarity heatmap color axis
        coloraxis={
            'colorbar_len': 0.5,
            'colorbar_x': 0.8,
            'colorbar_y': 0.8,
            'cmin': 0,
            'cmax': 1,
            'colorscale': 'Orrd_r'
        },
        # Correlation heatmaps color axis
        coloraxis2={
            'colorbar_len': 0.5,
            'colorbar_y': 0.2,
            'cmin': -1,
            'cmax': 1,
            'colorscale': ['#03AFF1', '#000036', '#01E0C9'],
        },
        # Sync the zoom and pan of the bottom 2 graphs
        yaxis3={'visible': False, 'matches': 'y2'},
        xaxis3={'matches': 'x2'},
        height=900,
        width=900,
    )

    fig.update_yaxes(autorange='reversed')

    return fig
