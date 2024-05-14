import numpy as np
import pandas as pd
import plotly.express as px

from sdmetrics.reports.single_table._properties import BaseSingleTableProperty
from sdmetrics.single_column import CategoryCoverage, RangeCoverage


class Coverage(BaseSingleTableProperty):
    """Coverage property class for single table.

    This property assesses data coverage between the real and synthetic data.
    A metric score is computed column-wise and the final score is the average over all columns.
    The ``RangeCoverage`` metric is used for numerical and datetime columns while the
    ``CategoryCoverage`` is used for categorical and boolean columns.
    The other column types are ignored by this property.
    """

    _num_iteration_case = 'column'
    _sdtype_to_metric = {
        'numerical': RangeCoverage,
        'datetime': RangeCoverage,
        'categorical': CategoryCoverage,
        'boolean': CategoryCoverage,
    }

    def _generate_details(self, real_data, synthetic_data, metadata, progress_bar=None):
        """Generate the _details dataframe for the column coverage property.

        Args:
            real_data (pandas.DataFrame):
                The real data.
            synthetic_data (pandas.DataFrame):
                The synthetic data.
            metadata (dict):
                The metadata of the table
            progress_bar (tqdm.tqdm or None):
                The progress bar to use. Defaults to None.

        Returns:
            pandas.DataFrame
        """
        column_names, metric_names, scores = [], [], []
        error_messages = []
        for column_name in metadata['columns']:
            sdtype = metadata['columns'][column_name]['sdtype']
            try:
                if sdtype in self._sdtype_to_metric:
                    metric = self._sdtype_to_metric[sdtype]
                    column_score = metric.compute(
                        real_data[column_name], synthetic_data[column_name]
                    )
                    error_message = None
                else:
                    continue

            except Exception as e:
                column_score = np.nan
                error_message = f'{type(e).__name__}: {e}'
            finally:
                if progress_bar:
                    progress_bar.update()

            column_names.append(column_name)
            metric_names.append(metric.__name__)
            scores.append(column_score)
            error_messages.append(error_message)

        result = pd.DataFrame({
            'Column': column_names,
            'Metric': metric_names,
            'Score': scores,
            'Error': error_messages,
        })

        if result['Error'].isna().all():
            result = result.drop('Error', axis=1)

        return result

    def get_visualization(self):
        """Create a plot to show the column coverage scores.

        Returns:
            plotly.graph_objects._figure.Figure
        """
        average_score = self._compute_average()

        fig = px.bar(
            data_frame=self.details.dropna(subset=['Score']),
            x='Column',
            y='Score',
            title=f'Data Diagnostics: Column Coverage (Average Score={round(average_score, 2)})',
            category_orders={'group': list(self.details['Column'])},
            color='Metric',
            color_discrete_map={
                'RangeCoverage': '#000036',
                'CategoryCoverage': '#03AFF1',
            },
            pattern_shape='Metric',
            pattern_shape_sequence=['', '/'],
            hover_name='Column',
            hover_data={
                'Column': False,
                'Metric': True,
                'Score': True,
            },
        )

        fig.update_yaxes(range=[0, 1], title_text='Diagnostic Score')

        fig.update_layout(
            xaxis_categoryorder='total ascending',
            plot_bgcolor='#F5F5F8',
            margin={'t': 150},
        )

        return fig
