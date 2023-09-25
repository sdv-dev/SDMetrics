"""Column pair trends property for multi-table."""
import itertools

import numpy as np
import pandas as pd
import plotly.express as px

from sdmetrics.reports.multi_table._properties import BaseMultiTableProperty
from sdmetrics.reports.single_table._properties import (
    ColumnPairTrends as SingleTableColumnPairTrends)
from sdmetrics.reports.utils import PlotConfig


class InterTableTrends(BaseMultiTableProperty):
    """Column pair trends property for multi-table.

    This property evaluates the matching in trends between pairs of real
    and synthetic data columns across related tables. Each pair's correlation is
    calculated and the final score represents the average of these measures across
    all column pairs
    """

    _num_iteration_case = 'inter_table_column_pair'

    def get_score(self, real_data, synthetic_data, metadata, progress_bar=None):
        """Get the average score of all the individual metric scores computed.

        Args:
            real_data (pandas.DataFrame):
                The real data.
            synthetic_data (pandas.DataFrame):
                The synthetic data.
            metadata (dict):
                The metadata, which contains each column's data type as well as relationships.
            progress_bar (tqdm.tqdm or None):
                The progress bar object. Defaults to None.

        Returns:
            float:
                The average score for the property for all the individual metric scores computed.
        """
        all_details = pd.DataFrame(columns=[
            'Parent Table',
            'Child Table',
            'Foreign Key',
            'Column 1',
            'Column 2',
            'Metric',
            'Score',
            'Real Correlation',
            'Synthetic Correlation',
            'Error'
        ])
        for relationship in metadata['relationships']:
            parent = relationship['parent_table_name']
            child = relationship['child_table_name']
            foreign_key = relationship['child_foreign_key']
            primary_key = relationship['parent_primary_key']

            real_parent = real_data[parent].add_prefix(f'{parent}.')
            real_child = real_data[child].add_prefix(f'{child}.')
            synthetic_parent = synthetic_data[parent].add_prefix(f'{parent}.')
            synthetic_child = synthetic_data[child].add_prefix(f'{child}.')

            child_index = f'{child}.{foreign_key}'
            parent_index = f'{parent}.{primary_key}'

            denormalized_real = real_child.merge(
                real_parent, left_on=child_index, right_on=parent_index
            )
            denormalized_synthetic = synthetic_child.merge(
                synthetic_parent, left_on=child_index, right_on=parent_index
            )

            child_meta = metadata['tables'][child]
            parent_meta = metadata['tables'][parent]
            merged_metadata = metadata['tables'][child].copy()
            child_cols = {
                f'{child}.{col}': col_meta for col, col_meta in child_meta['columns'].items()
            }
            parent_cols = {
                f'{parent}.{col}': col_meta for col, col_meta in parent_meta['columns'].items()
            }
            merged_metadata['columns'] = {**child_cols, **parent_cols}
            if 'primary_key' in merged_metadata:
                merged_metadata['primary_key'] = f'{child}.{merged_metadata["primary_key"]}'

            parent_child_pairs = itertools.product(parent_cols.keys(), child_cols.keys())

            self._properties[(parent, child, foreign_key)] = SingleTableColumnPairTrends()
            details = self._properties[(parent, child, foreign_key)]._generate_details(
                denormalized_real, denormalized_synthetic, merged_metadata,
                progress_bar=progress_bar, column_pairs=parent_child_pairs
            )
            details['Parent Table'] = parent
            details['Child Table'] = child
            details['Foreign Key'] = foreign_key
            details['Column 1'] = details['Column 1'].str.replace(f'{parent}.', '', n=1)
            details['Column 2'] = details['Column 2'].str.replace(f'{child}.', '', n=1)

            all_details = pd.concat([all_details, details]).reset_index(drop=True)

        if all_details['Error'].isna().all():
            all_details = all_details.drop('Error', axis=1)
        else:
            all_details['Error'] = all_details['Error'].replace({np.nan: None})

        self.details = all_details
        self.is_computed = True

        return self._compute_average()

    def get_visualization(self, table_name=None):
        """Create a plot to show the inter table trends data.

        Returns:
            plotly.graph_objects._figure.Figure

        Args:
            table_name (str, optional):
                Table to plot. Defaults to None.

        Raises:
            - ``ValueError`` if property has not been computed.

        Returns:
            plotly.graph_objects._figure.Figure
        """
        if not self.is_computed:
            raise ValueError(
                'The property must be computed before getting a visualization.'
                'Please call the ``get_score`` method first.'
            )

        to_plot = self.details.copy()
        if table_name is not None:
            to_plot = to_plot[
                (to_plot['Parent Table'] == table_name) |
                (to_plot['Child Table'] == table_name)
            ]

        parent_cols = to_plot['Parent Table'] + '.' + to_plot['Column 1']
        child_cols = to_plot['Child Table'] + '.' + to_plot['Column 2']
        to_plot['Columns'] = parent_cols + ', ' + child_cols
        duplicated = to_plot['Columns'].duplicated(keep=False)
        to_plot['Columns'][duplicated] = to_plot['Columns'][duplicated] + \
            ' (' + to_plot['Foreign Key'][duplicated] + ')'

        to_plot['Real Correlation'] = to_plot['Real Correlation'].fillna('None')
        to_plot['Synthetic Correlation'] = to_plot['Synthetic Correlation'].fillna('None')

        average_score = round(to_plot['Score'].mean(), 2)

        fig = px.bar(
            to_plot,
            x='Columns',
            y='Score',
            title=f'Data Quality: Column Shapes (Average Score={average_score})',
            category_orders={'group': to_plot['Columns']},
            color='Metric',
            color_discrete_map={
                'ContingencySimilarity': PlotConfig.DATACEBO_DARK,
                'CorrelationSimilarity': PlotConfig.DATACEBO_BLUE,
            },
            pattern_shape='Metric',
            pattern_shape_sequence=['', '/'],
            custom_data=[
                'Foreign Key',
                'Metric',
                'Score',
                'Real Correlation',
                'Synthetic Correlation'
            ]
        )

        fig.update_yaxes(range=[0, 1])

        fig.update_traces(
            hovertemplate='<br>'.join([
                '%{x}',
                '%{customdata[0]}',
                '',
                'Metric=%{customdata[1]}',
                'Score=%{customdata[2]}',
                'Real Correlation=%{customdata[3]}',
                'Synthetic Correlation=%{customdata[4]}<extra></extra>'
            ])
        )

        fig.update_layout(
            xaxis_categoryorder='total ascending',
            plot_bgcolor=PlotConfig.BACKGROUND_COLOR,
            margin={'t': 150},
            font={'size': PlotConfig.FONT_SIZE},
        )

        return fig
