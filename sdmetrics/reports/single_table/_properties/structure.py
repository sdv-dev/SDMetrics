import numpy as np
import pandas as pd
import plotly.express as px

from sdmetrics.reports.single_table._properties import BaseSingleTableProperty
from sdmetrics.single_table import TableFormat


class Structure(BaseSingleTableProperty):
    """Structure property class for single table.

    This property checks to see whether the overall structure of the synthetic
    data is the same as the real data.
    """

    _num_iteration_case = 'table'

    def _generate_details(self, real_data, synthetic_data, metadata, progress_bar=None):
        """Generate the _details dataframe for the structure property.

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
        column_to_ignore_dtype = []
        non_pii_sdtype = [
            'numerical', 'datetime', 'categorical', 'boolean'
        ]
        for column_name in metadata['columns']:
            sdtype = metadata['columns'][column_name]['sdtype']
            if sdtype in non_pii_sdtype:
                continue

            column_to_ignore_dtype.append(column_name)

        try:
            score = TableFormat.compute(
                real_data, synthetic_data,
                ignore_dtype_columns=column_to_ignore_dtype
            )
            error_message = None

        except Exception as e:
            score = np.nan
            error_message = f'{type(e).__name__}: {e}'

        finally:
            if progress_bar:
                progress_bar.update()

        result = pd.DataFrame({
            'Metric': 'TableFormat',
            'Score': score,
            'Error': error_message,
        }, index=[0])

        if result['Error'].isna().all():
            result = result.drop('Error', axis=1)

        return result

    def get_visualization(self):
        """Create a plot to show the structure property score.

        Returns:
            plotly.graph_objects._figure.Figure
        """
        average_score = self._compute_average()

        fig = px.bar(
            data_frame=self.details.dropna(subset=['Score']),
            x='Table',
            y='Score',
            title=f'Data Diagnostics: Structure (Average Score={round(average_score, 2)})',
            category_orders={'group': list(self.details['Column'])},
            color='Metric',
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
