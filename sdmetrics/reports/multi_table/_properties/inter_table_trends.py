"""Column pair trends property for multi-table."""

import itertools

import pandas as pd
import plotly.express as px

from sdmetrics.reports.multi_table._properties import BaseMultiTableProperty
from sdmetrics.reports.single_table._properties import (
    ColumnPairTrends as SingleTableColumnPairTrends,
)
from sdmetrics.reports.utils import PlotConfig


class InterTableTrends(BaseMultiTableProperty):
    """Column pair trends property for multi-table.

    This property evaluates the matching in trends between pairs of real
    and synthetic data columns across related tables. Each pair's correlation is
    calculated and the final score represents the average of these measures across
    all column pairs
    """

    _num_iteration_case = 'inter_table_column_pair'

    def _denormalize_tables(self, real_data, synthetic_data, relationship):
        """Merge a parent and child table into one denormalized table.

        Args:
            real_data (dict[str, pandas.DataFrame]):
                The real data.
            synthetic_data (dict[str, pandas.DataFrame]):
                The synthetic data.
            relationship (dict):
                The relationship to denormalize.

        Returns:
            tuple(pd.DataFrame, pd.DataFrame)
                The denormalized real table and the denormalized synthetic table.
        """
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

        return denormalized_real, denormalized_synthetic

    def _merge_metadata(self, metadata, parent_table, child_table):
        """Merge the metadata of a parent and child table.

        Merge the metadata for two tables together. Column names will be prefixed
        with ``'{table_name}.'``. The merged table will inherit the child table's
        primary key.

        Args:
            metadata (dict):
                The metadata for the multi-table data.
            relationship (dict):
                The relationship to use to denormalize the metadata.

        Returns:
            (dict, list, list)
                The metadata dictionary for the merged table. Also returns the list of columns
                that came from the parent table, and the list of columns that came from the
                child table.
        """
        child_meta = metadata['tables'][child_table]
        parent_meta = metadata['tables'][parent_table]
        merged_metadata = metadata['tables'][child_table].copy()
        child_cols = {
            f'{child_table}.{col}': col_meta for col, col_meta in child_meta['columns'].items()
        }
        parent_cols = {
            f'{parent_table}.{col}': col_meta for col, col_meta in parent_meta['columns'].items()
        }
        merged_metadata['columns'] = {**child_cols, **parent_cols}
        if 'primary_key' in merged_metadata:
            primary_key = merged_metadata['primary_key']
            merged_metadata['primary_key'] = f'{child_table}.{primary_key}'

        return merged_metadata, list(parent_cols.keys()), list(child_cols.keys())

    def _generate_details(self, real_data, synthetic_data, metadata, progress_bar=None):
        """Generate the details dataframe for the metric.

        Args:
            real_data (dict[str, pandas.DataFrame]):
                The real data.
            synthetic_data (dict[str, pandas.DataFrame]):
                The synthetic data.
            metadata (dict):
                The metadata, which contains each column's data type as well as relationships.
            progress_bar (tqdm.tqdm or None):
                The progress bar object. Defaults to None.
        """
        all_details = []
        for relationship in metadata.get('relationships', []):
            parent = relationship['parent_table_name']
            child = relationship['child_table_name']
            foreign_key = relationship['child_foreign_key']

            denormalized_real, denormalized_synthetic = self._denormalize_tables(
                real_data, synthetic_data, relationship
            )

            merged_metadata, parent_cols, child_cols = self._merge_metadata(metadata, parent, child)

            parent_child_pairs = itertools.product(parent_cols, child_cols)

            self._properties[(parent, child, foreign_key)] = SingleTableColumnPairTrends()
            details = self._properties[(parent, child, foreign_key)]._generate_details(
                denormalized_real,
                denormalized_synthetic,
                merged_metadata,
                progress_bar=progress_bar,
                column_pairs=parent_child_pairs,
            )

            details['Parent Table'] = parent
            details['Child Table'] = child
            details['Foreign Key'] = foreign_key
            if not details.empty:
                details['Column 1'] = details['Column 1'].str.replace(
                    f'{parent}.', '', n=1, regex=False
                )
                details['Column 2'] = details['Column 2'].str.replace(
                    f'{child}.', '', n=1, regex=False
                )
            all_details.append(details)

        if len(all_details) > 0:
            self.details = pd.concat(all_details, axis=0).reset_index(drop=True)
            detail_columns = [
                'Parent Table',
                'Child Table',
                'Foreign Key',
                'Column 1',
                'Column 2',
                'Metric',
                'Score',
                'Real Correlation',
                'Synthetic Correlation',
            ]
            if 'Error' in self.details.columns:
                detail_columns.append('Error')

            self.details = self.details[detail_columns]

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
                (to_plot['Parent Table'] == table_name) | (to_plot['Child Table'] == table_name)
            ]

        parent_cols = to_plot['Parent Table'] + '.' + to_plot['Column 1']
        child_cols = to_plot['Child Table'] + '.' + to_plot['Column 2']
        to_plot['Columns'] = parent_cols + ', ' + child_cols
        duplicated = to_plot['Columns'].duplicated(keep=False)
        to_plot['Columns'][duplicated] = (
            to_plot['Columns'][duplicated] + ' (' + to_plot['Foreign Key'][duplicated] + ')'
        )

        to_plot['Real Correlation'] = to_plot['Real Correlation'].fillna('None')
        to_plot['Synthetic Correlation'] = to_plot['Synthetic Correlation'].fillna('None')

        average_score = round(to_plot['Score'].mean(), 2)

        fig = px.bar(
            to_plot,
            x='Columns',
            y='Score',
            title=f'Data Quality: Intertable Trends (Average Score={average_score})',
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
                'Synthetic Correlation',
            ],
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
                'Synthetic Correlation=%{customdata[4]}<extra></extra>',
            ])
        )

        fig.update_layout(
            xaxis_categoryorder='total ascending',
            plot_bgcolor=PlotConfig.BACKGROUND_COLOR,
            margin={'t': 150},
            font={'size': PlotConfig.FONT_SIZE},
        )

        return fig
