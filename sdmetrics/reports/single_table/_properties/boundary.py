import numpy as np
import pandas as pd
import plotly.express as px

from sdmetrics.errors import InvalidDataError
from sdmetrics.reports.single_table._properties import BaseSingleTableProperty
from sdmetrics.single_column import BoundaryAdherence


class Boundary(BaseSingleTableProperty):
    """Boundary property class for single table.

    This property assesses the boundary adherence of the synthetic data over the real data.
    The BoundaryAdherence metric is computed column-wise and the final score is the average
    over all columns. This metric is computed over numerical and datetime columns only.
    The other column types are ignored by this property.
    """

    metric = BoundaryAdherence
    _num_iteration_case = 'column'

    def _generate_details(self, real_data, synthetic_data, metadata, progress_bar=None):
        """Generate the _details dataframe for the boundary property.

        Args:
            real_data (pandas.DataFrame):
                The real data.
            synthetic_data (pandas.DataFrame):
                The synthetic data.
            metadata (dict):
                The metadata of the table.
            progress_bar (tqdm.tqdm or None):
                The progress bar to use. Defaults to None.

        Returns:
            pandas.DataFrame
        """
        column_names, metric_names, scores = [], [], []
        error_messages = []
        for column_name in metadata['columns']:
            sdtype = metadata['columns'][column_name]['sdtype']
            error_message = None
            try:
                if sdtype in ('numerical', 'datetime'):
                    real_column = real_data[column_name]
                    synthetic_column = synthetic_data[column_name]
                    real_column_is_na = all(real_column.isna())
                    synthetic_column_is_na = all(synthetic_column.isna())

                    if real_column_is_na and synthetic_column_is_na:
                        raise InvalidDataError('All NaN values in both real and synthetic data.')
                    elif real_column_is_na:
                        raise InvalidDataError('All NaN values in real data.')
                    elif synthetic_column_is_na:
                        raise InvalidDataError('All NaN values in synthetic data.')
                    else:
                        column_score = self.metric.compute(real_column, synthetic_column)

                else:
                    continue

            except Exception as e:
                column_score = np.nan
                error_message = f'{type(e).__name__}: {e}'
            finally:
                if progress_bar:
                    progress_bar.update()

            column_names.append(column_name)
            metric_names.append(self.metric.__name__)
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
        """Create a plot to show the column boundary scores.

        Returns:
            plotly.graph_objects._figure.Figure.
        """
        average_score = self._compute_average()

        fig = px.bar(
            data_frame=self.details,
            x='Column',
            y='Score',
            title=f'Data Diagnostics: Column Boundary (Average Score={round(average_score, 2)})',
            category_orders={'group': list(self.details['Column'])},
            color='Metric',
            color_discrete_map={
                'BoundaryAdherence': '#000036',
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
