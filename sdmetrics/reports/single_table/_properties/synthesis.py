import numpy as np
import pandas as pd
import plotly.express as px

from sdmetrics.reports.single_table._properties import BaseSingleTableProperty
from sdmetrics.single_table import NewRowSynthesis


class Synthesis(BaseSingleTableProperty):
    """Synthesis property class for single table.

    This property assesses the novelty of the synthetic data over the real data.
    The ``NewRowSynthesis`` metric is computed over the real and synthetic table to
    score the proportion of new rows in the synthetic data.
    """

    _num_iteration_case = 'table'
    metric = NewRowSynthesis

    def _generate_details(self, real_data, synthetic_data, metadata, progress_bar=None):
        """Generate the _details dataframe for the synthesis property.

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
            pandas.DataFrame.
        """
        name = self.metric.__name__
        error_message = None

        sample_size = len(synthetic_data) if len(synthetic_data) < 10000 else 10000
        try:
            score_breakdown = self.metric.compute_breakdown(
                real_data=real_data,
                synthetic_data=synthetic_data,
                metadata=metadata,
                synthetic_sample_size=sample_size,
            )
            score = score_breakdown['score']
            num_matched_rows = score_breakdown['num_matched_rows']
            num_new_rows = score_breakdown['num_new_rows']

        except Exception as e:
            score = np.nan
            num_matched_rows = np.nan
            num_new_rows = np.nan
            error_message = f'{type(e).__name__}: {e}'

        finally:
            if progress_bar:
                progress_bar.update()

        result = pd.DataFrame(
            {
                'Metric': name,
                'Score': score,
                'Num Matched Rows': num_matched_rows,
                'Num New Rows': num_new_rows,
                'Error': error_message,
            },
            index=[0],
        )

        if pd.isna(result['Error'].iloc[0]):
            result = result.drop('Error', axis=1)

        return result

    def get_visualization(self):
        """Create a plot to show the synthesis property.

        Returns:
            plotly.graph_objects._figure.Figure.
        """
        labels = ['Exact Matches', 'Novel Rows']
        values = list(self.details[['Num Matched Rows', 'Num New Rows']].iloc[0])

        average_score = round(self._compute_average(), 2)

        fig = px.pie(
            values=values,
            names=labels,
            color=['Exact Matches', 'Novel Rows'],
            color_discrete_map={'Exact Matches': '#F16141', 'Novel Rows': '#36B37E'},
            hole=0.4,
            title=f'Data Diagnostic: Synthesis (Score={average_score})',
        )

        fig.update_traces(hovertemplate='<b>%{label}</b><br>%{value} rows')

        return fig
