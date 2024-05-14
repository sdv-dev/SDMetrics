"""Structure property for multi-table."""

import plotly.express as px

from sdmetrics.errors import VisualizationUnavailableError
from sdmetrics.reports.multi_table._properties import BaseMultiTableProperty
from sdmetrics.reports.single_table._properties import Structure as SingleTableStructure
from sdmetrics.reports.utils import PlotConfig


class Structure(BaseMultiTableProperty):
    """Structure property class for multi-table.

    This property checks to see whether the overall structure of the synthetic
    data is the same as the real data. The property is calculated for each table.
    """

    _single_table_property = SingleTableStructure
    _num_iteration_case = 'table'

    def get_visualization(self, table_name=None):
        """Return a visualization for each score in the property.

        Args:
            table_name:
                If a table name is provided, an error is raised.

        Returns:
            plotly.graph_objects._figure.Figure
                The visualization for the property.
        """
        if table_name:
            raise VisualizationUnavailableError(
                'The Structure property does not have a supported visualization for'
                ' individual tables.'
            )

        average_score = self._compute_average()
        fig = px.bar(
            data_frame=self.details,
            x='Table',
            y='Score',
            title=f'Data Diagnostic: Structure (Average Score={average_score})',
            category_orders={'group': list(self.details['Table'])},
            color='Metric',
            color_discrete_map={
                'TableStructure': PlotConfig.DATACEBO_DARK,
            },
            pattern_shape='Metric',
            pattern_shape_sequence=[''],
            hover_name='Table',
            hover_data={
                'Table': False,
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
