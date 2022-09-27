"""Utility methods for plotting metric scores."""

import numpy as np
import pandas as pd
import plotly.express as px

BAR_COLOR = '#000036'
BACKGROUND_COLOR = '#F5F5F8'


def _get_table_relationships_data(score_breakdowns):
    """Convert the score breakdowns into the desired table relationships data format.

    Args:
        score_breakdowns (dict):
            A mapping of parent child relationship metrics to the score breakdowns.

    Returns:
        pandas.DataFrame
    """
    relationships = []
    metrics = []
    scores = []

    for metric, score_breakdown in score_breakdowns.items():
        for tables, result in score_breakdown.items():
            if not np.isnan(result['score']):
                relationships.append(f'{tables[1]} → {tables[0]}')
                metrics.append(metric)
                scores.append(result['score'])

    return pd.DataFrame({
        'Child → Parent Relationship': relationships,
        'Metric': metrics,
        'Quality Score': scores,
    })


def get_table_relationships_plot(score_breakdowns):
    """Get the table relationships plot from the parent child relationship scores for a table.

    Args:
        score_breakdowns (dict):
            The parent child relationship scores.

    Returns:
        plotly.graph_objects._figure.Figure
    """
    plot_data = _get_table_relationships_data(score_breakdowns)
    average_score = round(plot_data['Quality Score'].mean(), 2)

    fig = px.bar(
        plot_data,
        x='Child → Parent Relationship',
        y='Quality Score',
        title=f'Data Quality: Table Relationships (Average Score={average_score})',
        color='Metric',
        color_discrete_sequence=[BAR_COLOR],
        hover_name='Child → Parent Relationship',
        hover_data={
            'Child → Parent Relationship': False,
            'Metric': True,
            'Quality Score': True,
        },
    )

    fig.update_yaxes(range=[0, 1])

    fig.update_layout(
        xaxis_categoryorder='total ascending',
        plot_bgcolor=BACKGROUND_COLOR,
    )

    return fig
