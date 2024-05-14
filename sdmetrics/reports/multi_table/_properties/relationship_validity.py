import numpy as np
import pandas as pd
import plotly.express as px

from sdmetrics.column_pairs.statistical import CardinalityBoundaryAdherence, ReferentialIntegrity
from sdmetrics.reports.multi_table._properties.base import BaseMultiTableProperty
from sdmetrics.reports.utils import PlotConfig


class RelationshipValidity(BaseMultiTableProperty):
    """``Relationship Validity`` property.

    This property measures the validity of the relationship
    from the primary key and the foreign key perspective.

    """

    _num_iteration_case = 'relationship'

    def _generate_details(self, real_data, synthetic_data, metadata, progress_bar=None):
        """Generate the _details dataframe for the relationship validity property.

        Args:
            real_data (dict[str, pandas.DataFrame]):
                The tables from the real dataset, passed as a dictionary of
                table names and pandas.DataFrames.
            synthetic_data (dict[str, pandas.DataFrame]):
                The tables from the synthetic dataset, passed as a dictionary of
                table names and pandas.DataFrames.
            metadata (dict):
                The metadata, which contains each column's data type as well as relationships.
            progress_bar (tqdm.tqdm or None):
                The progress bar object. Defaults to ``None``.

        Returns:
            float:
                The average score for the property for all the individual metric scores computed.
        """
        child_tables, parent_tables = [], []
        primary_key, foreign_key = [], []
        metric_names, scores, error_messages = [], [], []
        metrics = [ReferentialIntegrity, CardinalityBoundaryAdherence]
        for relation in metadata.get('relationships', []):
            real_columns = self._extract_tuple(real_data, relation)
            synthetic_columns = self._extract_tuple(synthetic_data, relation)
            for metric in metrics:
                try:
                    relation_score = metric.compute(
                        real_columns,
                        synthetic_columns,
                    )
                    error_message = None
                except Exception as e:
                    relation_score = np.nan
                    error_message = f'{type(e).__name__}: {e}'

                child_tables.append(relation['child_table_name'])
                parent_tables.append(relation['parent_table_name'])
                primary_key.append(relation['parent_primary_key'])
                foreign_key.append(relation['child_foreign_key'])
                metric_names.append(metric.__name__)
                scores.append(relation_score)
                error_messages.append(error_message)

            if progress_bar:
                progress_bar.update()

        self.details = pd.DataFrame({
            'Parent Table': parent_tables,
            'Child Table': child_tables,
            'Primary Key': primary_key,
            'Foreign Key': foreign_key,
            'Metric': metric_names,
            'Score': scores,
            'Error': error_messages,
        })

    def _get_table_relationships_plot(self, table_name):
        """Get the table relationships plot from the parent child relationship scores for a table.

        Args:
            table_name (str):
                Table name to get details table for.

        Returns:
            plotly.graph_objects._figure.Figure
        """
        plot_data = self.get_details(table_name).copy()
        column_name = 'Child → Parent Relationship'
        plot_data[column_name] = (
            plot_data['Child Table']
            + ' ('
            + plot_data['Foreign Key']
            + ') → '
            + plot_data['Parent Table']
        )
        plot_data = plot_data.drop(['Child Table', 'Parent Table'], axis=1)

        average_score = round(plot_data['Score'].mean(), 2)

        fig = px.bar(
            plot_data,
            x='Child → Parent Relationship',
            y='Score',
            title=f'Data Diagnostic: Relationship Validity (Average Score={average_score})',
            color='Metric',
            color_discrete_sequence=[PlotConfig.DATACEBO_DARK, PlotConfig.DATACEBO_GREEN],
            pattern_shape='Metric',
            pattern_shape_sequence=['', '/'],
            hover_name='Child → Parent Relationship',
            hover_data={
                'Child → Parent Relationship': False,
                'Metric': True,
                'Score': True,
            },
            barmode='group',
        )

        fig.update_yaxes(range=[0, 1])

        fig.update_layout(
            xaxis_categoryorder='total ascending',
            plot_bgcolor=PlotConfig.BACKGROUND_COLOR,
            font={'size': PlotConfig.FONT_SIZE},
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
