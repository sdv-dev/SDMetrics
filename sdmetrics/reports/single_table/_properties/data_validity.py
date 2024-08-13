import numpy as np
import pandas as pd
import plotly.express as px

from sdmetrics.reports.single_table._properties import BaseSingleTableProperty
from sdmetrics.reports.utils import PlotConfig
from sdmetrics.single_column import BoundaryAdherence, CategoryAdherence, KeyUniqueness


class DataValidity(BaseSingleTableProperty):
    """Data Validity property class for single table.

    This property computes, at base, whether each column contains valid data.
    The metric is based on the type data in each column.
    The BoundaryAdherence metric is used for numerical and datetime columns, the CategoryAdherence
    is used for categorical and boolean columns and the KeyUniqueness for primary
    and alternate keys. The other column types are ignored by this property.
    """

    _num_iteration_case = 'column'
    _sdtype_to_metric = {
        'numerical': BoundaryAdherence,
        'datetime': BoundaryAdherence,
        'categorical': CategoryAdherence,
        'boolean': CategoryAdherence,
    }

    def _generate_details(self, real_data, synthetic_data, metadata, progress_bar=None):
        """Generate the _details dataframe for the data validity property.

        Args:
            real_data (pandas.DataFrame):
                The real data
            synthetic_data (pandas.DataFrame):
                The synthetic data
            metadata (dict):
                The metadata of the table
            progress_bar (tqdm.tqdm or None):
                The progress bar to use. Defaults to None.
        """
        column_names, metric_names, scores = [], [], []
        error_messages = []
        primary_key = metadata.get('primary_key')
        alternate_keys = metadata.get('alternate_keys', [])
        for column_name in metadata['columns']:
            sdtype = metadata['columns'][column_name]['sdtype']
            primary_key_match = column_name == primary_key
            alternate_key_match = column_name in alternate_keys
            is_unique = primary_key_match or alternate_key_match

            try:
                if sdtype not in self._sdtype_to_metric and not is_unique:
                    continue

                metric = self._sdtype_to_metric.get(sdtype, KeyUniqueness)
                column_score = metric.compute(real_data[column_name], synthetic_data[column_name])
                error_message = None

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
        """Create a plot to show the data validity scores.

        Returns:
            plotly.graph_objects._figure.Figure
        """
        average_score = round(self._compute_average(), 2)

        fig = px.bar(
            data_frame=self.details,
            x='Column',
            y='Score',
            title=f'Data Diagnostic: Data Validity (Average Score={average_score})',
            category_orders={'group': list(self.details['Column'])},
            color='Metric',
            color_discrete_map={
                'BoundaryAdherence': PlotConfig.DATACEBO_DARK,
                'CategoryAdherence': PlotConfig.DATACEBO_BLUE,
                'KeyUniqueness': PlotConfig.DATACEBO_GREEN,
            },
            pattern_shape='Metric',
            pattern_shape_sequence=['', '/', '.'],
            hover_name='Column',
            hover_data={
                'Column': False,
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
