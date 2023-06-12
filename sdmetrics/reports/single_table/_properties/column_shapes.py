import warnings

import numpy as np
import pandas as pd
import plotly.express as px
import tqdm

from sdmetrics.reports.single_table._properties import BaseSingleTableProperty
from sdmetrics.single_column.statistical.kscomplement import KSComplement
from sdmetrics.single_column.statistical.tv_complement import TVComplement


class ColumnShapes(BaseSingleTableProperty):
    """Column Shapes property class for single table."""

    metrics = [KSComplement, TVComplement]
    _sdtype_to_metric = {
        'numerical': KSComplement,
        'datetime': KSComplement,
        'categorical': TVComplement,
        'boolean': TVComplement
    }

    def _generate_details(self, real_data, synthetic_data, metadata, progress_bar=tqdm.tqdm):
        """Generate the _details dataframe for the column shapes property.

        Args:
            real_data (pandas.DataFrame):
                The real data
            synthetic_data (pandas.DataFrame):
                The synthetic data
            metadata (dict):
                The metadata of the table
            progress_bar:
                The progress bar to use. Defaults to tqdm.
        """
        column_names, metric_names, scores = [], [], []
        for column_name in progress_bar(metadata['columns']):
            sdtype = metadata['columns'][column_name]['sdtype']
            try:
                if sdtype in self._sdtype_to_metric:
                    metric = self._sdtype_to_metric[sdtype]
                    column_score = metric.compute(
                        real_data[column_name], synthetic_data[column_name]
                    )
                else:
                    continue

            except Exception as e:
                column_score = np.nan
                warnings.warn(
                        f"Unable to compute Column Shape for column '{column_name}'. "
                        f'Encountered Error: {type(e).__name__} {e}'
                )

            column_names.append(column_name)
            metric_names.append(metric.__name__)
            scores.append(column_score)

        result = pd.DataFrame({
            'Column name': column_names,
            'Metric': metric_names,
            'Score': scores,
        })

        return result

    def get_visualization(self):
        """Create a plot to show the column shape scores.

        Returns:
            plotly.graph_objects._figure.Figure
        """
        average_score = self._compute_average()

        fig = px.bar(
            self._details,
            x='Column name',
            y='Score',
            title=f'Data Quality: Column Shapes (Average Score={average_score})',
            category_orders={'group': self._details['Column name']},
            color='Metric',
            color_discrete_map={
                'KSComplement': '#000036',
                'TVComplement': '#03AFF1',
            },
            pattern_shape='Metric',
            pattern_shape_sequence=['', '/'],
            hover_name='Column name',
            hover_data={
                'Column name': False,
                'Metric': True,
                'Score': True,
            },
        )

        fig.update_yaxes(range=[0, 1])

        fig.update_layout(
            xaxis_categoryorder='total ascending',
            plot_bgcolor='#F5F5F8',
            margin={'t': 150},
        )

        return fig
