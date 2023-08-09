"""Utility methods for plotting metric scores."""
import plotly.express as px

BAR_COLOR = '#000036'
BACKGROUND_COLOR = '#F5F5F8'


def _get_table_relationships_data(details_property):
    """Convert the score breakdowns into the desired table relationships data format.

    Args:
        details_property (pandas.DataFrame):
            The details property table.

    Returns:
        pandas.DataFrame
    """
    result = details_property.copy()
    result['Child → Parent Relationship'] = result['Child Table'] + ' → ' + result['Parent Table']
    result = result.drop(['Child Table', 'Parent Table'], axis=1)

    return result


def get_table_relationships_plot(details_property):
    """Get the table relationships plot from the parent child relationship scores for a table.

    Args:
        details_property (pandas.DataFrame):
            The details property table.

    Returns:
        plotly.graph_objects._figure.Figure
    """
    plot_data = _get_table_relationships_data(details_property)
    average_score = round(plot_data['Score'].mean(), 2)

    fig = px.bar(
        plot_data,
        x='Child → Parent Relationship',
        y='Score',
        title=f'Data Quality: Table Relationships (Average Score={average_score})',
        color='Metric',
        color_discrete_sequence=[BAR_COLOR],
        hover_name='Child → Parent Relationship',
        hover_data={
            'Child → Parent Relationship': False,
            'Metric': True,
            'Score': True,
        },
    )

    fig.update_yaxes(range=[0, 1])

    fig.update_layout(
        xaxis_categoryorder='total ascending',
        plot_bgcolor=BACKGROUND_COLOR,
    )

    return fig
