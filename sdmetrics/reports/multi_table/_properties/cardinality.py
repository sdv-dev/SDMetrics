import numpy as np
import pandas as pd
import plotly.express as px

from sdmetrics.multi_table.statistical import CardinalityShapeSimilarity
from sdmetrics.reports.multi_table._properties.base import BaseMultiTableProperty
from sdmetrics.reports.utils import PlotConfig


class Cardinality(BaseMultiTableProperty):
    """``Cardinality`` class.

    Property that uses ``sdmetrics.multi_table.statistical.CardinalityShapeSimilarity`` metric
    in order to compute and plot the scores of cardinality shape similarity in the given tables.
    """

    _num_iteration_case = 'relationship'

    def _generate_details(self, real_data, synthetic_data, metadata, progress_bar=None):
        """Get the average score of cardinality shape similarity in the given tables.

        Args:
            real_data (pandas.DataFrame):
                The real data.
            synthetic_data (pandas.DataFrame):
                The synthetic data.
            metadata (dict):
                The metadata, which contains each column's data type as well as relationships.
            progress_bar (tqdm.tqdm or None):
                The progress bar object. Defaults to ``None``.

        Returns:
            float:
                The average score for the property for all the individual metric scores computed.
        """
        child_tables, parent_tables = [], []
        metric_names, scores, error_messages = [], [], []
        for relation in metadata.get('relationships', []):
            relationships_metadata = {'relationships': [relation]}
            try:
                relation_score = CardinalityShapeSimilarity.compute(
                    real_data,
                    synthetic_data,
                    relationships_metadata
                )
                error_message = None
            except Exception as e:
                relation_score = np.nan
                error_message = f'{type(e).__name__}: {e}'
            finally:
                if progress_bar is not None:
                    progress_bar.update()

            child_tables.append(relation['child_table_name'])
            parent_tables.append(relation['parent_table_name'])
            metric_names.append('CardinalityShapeSimilarity')
            scores.append(relation_score)
            error_messages.append(error_message)

        self.details = pd.DataFrame({
            'Child Table': child_tables,
            'Parent Table': parent_tables,
            'Metric': metric_names,
            'Score': scores,
            'Error': error_messages,
        })

    def _get_details_for_table_name(self, table_name):
        """Return the details for the given table name.

        Args:
            table_name (str):
                Table name to get the details for.

        Returns:
            pandas.DataFrame:
                The details for the given table name.
        """
        is_child = self.details['Child Table'] == table_name
        is_parent = self.details['Parent Table'] == table_name
        return self.details[is_child | is_parent].copy()

    def get_details(self, table_name=None):
        """Return the details for the property.

        Args:
            table_name (str):
                Table name to get the details for.
                Defaults to ``None``.

        Returns:
            pandas.DataFrame:
                The details for the property.
        """
        if table_name is None:
            return self.details.copy()

        return self._get_details_for_table_name(table_name)

    def _get_table_relationships_plot(self, table_name):
        """Get the table relationships plot from the parent child relationship scores for a table.

        Args:
            table_name (str):
                Table name to get details table for.

        Returns:
            plotly.graph_objects._figure.Figure
        """
        plot_data = self._get_details_for_table_name(table_name).copy()
        column_name = 'Child → Parent Relationship'
        plot_data[column_name] = plot_data['Child Table'] + ' → ' + plot_data['Parent Table']
        plot_data = plot_data.drop(['Child Table', 'Parent Table'], axis=1)

        average_score = round(plot_data['Score'].mean(), 2)

        fig = px.bar(
            plot_data,
            x='Child → Parent Relationship',
            y='Score',
            title=f'Table Relationships (Average Score={average_score})',
            color='Metric',
            color_discrete_sequence=[PlotConfig.DATACEBO_DARK],
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
            plot_bgcolor=PlotConfig.BACKGROUND_COLOR,
            font={'size': PlotConfig.FONT_SIZE}
        )

        return fig

    def get_visualization(self, table_name):
        """Return a visualization for each score in the property.

        Args:
            table_name (str):
                Table name to get the visualization for.

        Returns:
            plotly.graph_objects._figure.Figure
                The visualization for the property.
        """
        return self._get_table_relationships_plot(table_name)
