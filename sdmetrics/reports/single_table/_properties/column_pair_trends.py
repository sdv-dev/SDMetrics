import warnings

import numpy as np
import pandas as pd
import plotly.express as px
import tqdm

from scipy.stats import pearsonr, spearmanr

from sdmetrics.reports.single_table._properties import BaseSingleTableProperty
from sdmetrics.column_pairs.statistical.correlation_similarity import CorrelationSimilarity
from sdmetrics.column_pairs.statistical.contingency_similarity import ContingencySimilarity
from sdmetrics.reports.utils import discretize_table_data


class ColumnPairTrends(BaseSingleTableProperty):

    metrics = [CorrelationSimilarity, ContingencySimilarity]
    _sdtype_to_shape = {
        'numerical': 'continuous',
        'datetime': 'continuous',
        'categorical': 'discrete',
        'boolean': 'discrete'
    }

    def datetime_to_numeric(self, data, metadata):
        data = data.copy()
        for column_name in metadata['columns']:
            sdtype = metadata['columns'][column_name]['sdtype']
            if sdtype == 'datetime':
                datetime_format = metadata['columns'][column_name].get('datetime_format', None)
                data[column_name] = pd.to_datetime(
                    data[column_name], format=datetime_format
                ).astype(np.int64)

        return data

    def _generate_details(self, real_data, synthetic_data, metadata, progress_bar=tqdm.tqdm):

        real_data = self.datetime_to_numeric(real_data, metadata)
        synthetic_data = self.datetime_to_numeric(synthetic_data, metadata)
        discrete_real, discrete_synthetic, _ = discretize_table_data(
            real_data, synthetic_data, metadata
        )
        print(real_data.dtypes)
        print(synthetic_data.dtypes)
        column_names_1 = []
        column_names_2 = []
        metric_names = []
        scores = []
        real_correlations = []
        synthetic_correaltions = []
        list_col_names = list(metadata['columns'])
        idx = 1
        for column_name_1 in progress_bar(list_col_names[:-1]):
            sdtype_col_1 = metadata['columns'][column_name_1]['sdtype']
            for column_name_2 in list_col_names[idx:]:
                sdtype_col_2 = metadata['columns'][column_name_2]['sdtype']
                try:
                    if sdtype_col_1 in self._sdtype_to_shape and sdtype_col_2 in self._sdtype_to_shape:
                        if self._sdtype_to_shape[sdtype_col_1] == self._sdtype_to_shape[sdtype_col_1]:
                            columns_real = real_data[[column_name_1, column_name_2]]
                            columns_synthetic = synthetic_data[[column_name_1, column_name_2]]

                            if self._sdtype_to_shape[sdtype_col_1] == 'continuous':
                                metric = CorrelationSimilarity
                            else:
                                metric = ContingencySimilarity
                        else:
                            metric = ContingencySimilarity
                            if self._sdtype_to_shape[sdtype_col_1] == 'continuous':
                                columns_real = pd.concat(
                                    [discrete_real[column_name_1], real_data[column_name_2]], axis=1
                                )
                                columns_synthetic = pd.concat(
                                    [discrete_synthetic[column_name_1], synthetic_data[column_name_2]],
                                    axis=1
                                )
                            else:
                                columns_real = pd.concat(
                                    [real_data[column_name_1], discrete_real[column_name_2]],
                                    axis=1
                                )
                                columns_synthetic = pd.concat(
                                    [synthetic_data[column_name_1], discrete_synthetic[column_name_2]],
                                    axis=1
                                )

                    pair_score = metric.compute(
                        real_data=columns_real, synthetic_data=columns_synthetic
                    )
                except Exception as e:
                    pair_score = np.nan
                    warnings.warn(
                            f"Unable to compute Column Pair Trends for column ('{column_name_1}', "
                            f"'{column_name_2}'). Encountered Error: {type(e).__name__} {e}"
                    )

                try:
                    real_correlation = pearsonr(
                        real_data[column_name_1], real_data[column_name_2])[0]
                except Exception as e:
                    real_correlation = np.nan

                try:
                    synthetic_correlation = pearsonr(
                        synthetic_data[column_name_1], synthetic_data[column_name_2])[0]
                except Exception as e:
                    synthetic_correlation = np.nan

                column_names_1.append(column_name_1)
                column_names_2.append(column_name_2)
                metric_names.append(metric.__name__)
                scores.append(pair_score)
                real_correlations.append(real_correlation)
                synthetic_correaltions.append(synthetic_correlation)

            idx+=1

        result = pd.DataFrame({
            'Column name 1': column_names_1,
            'Column name 2': column_names_2,
            'Metric': metric_names,
            'Score': scores,
            'Real correlation': real_correlations,
            'Synthetic correlation': synthetic_correaltions
        })

        return result

    def _get_correlation_matrix(self, column_name):

        if column_name not in ['Score', 'Real correlation', 'Synthetic correlation']:
            raise ValueError(f"Invalid column name: {column_name}")
        
        pivoted = self._details.pivot(index='Column 1', columns='Column 2', values=column_name)

        symmetric = pivoted.fillna(0) + pivoted.fillna(0).T

        for i in range(len(symmetric)):
            symmetric.iloc[i,i] = 1

        return symmetric

    def _get_similarity_correlation_matrix(self):
        """Convert the _details scores to a similiarity correlation matrix.

        Returns:
            pandas.DataFrame
        """
        column_names = self._details['Column name 1'].unique()
        
        similarity_correlation = pd.DataFrame(
            index=column_names,
            columns=column_names,
            dtype='float',
        )
        np.fill_diagonal(similarity_correlation.to_numpy(), 1.0)

        for column_name_1, column_name_2 in zip(column_names, column_names[1:]):
            rows_col_name_1 = self._details['Column name 1'] == column_name_1
            rows_col_name_2 = self._details['Column name 2'] == column_name_2
            score = self._details.loc[rows_col_name_1 * rows_col_name_2]['Score']

            similarity_correlation.loc[column_name_1, column_name_2] = score
            similarity_correlation.loc[column_name_2, column_name_1] = score

        return similarity_correlation


    def get_visualization(self):
        """Create a plot to show the column pairs data.

        This plot will have one graph in the top row and two in the bottom row.

        Args:
            score_breakdowns (dict):
                The score breakdowns of the column pairs metric scores.
            average_score (float):
                The average score. If None, the average score will be computed from
                ``score_breakdowns``.

        Returns:
            plotly.graph_objects._figure.Figure
        """
        all_columns = []
        all_scores = []
        for _, score_breakdown in score_breakdowns.items():
            for column_pair, result in score_breakdown.items():
                all_columns.append(column_pair[0])
                all_columns.append(column_pair[1])
                all_scores.append(result['score'])

        if average_score is None:
            average_score = np.mean(all_scores)

        similarity_correlation = self._get_similarity_correlation_matrix()
        real_correlation, synthetic_correlation = _get_numerical_correlation_matrices(score_breakdowns)

        fig = make_subplots(
            rows=2,
            cols=2,
            subplot_titles=[
                'Real vs. Synthetic Similarity',
                'Numerical Correlation (Real Data)',
                'Numerical Correlation (Synthetic Data)',
            ],
            specs=[[{'colspan': 2, 'l': 0.26, 'r': 0.26}, None], [{}, {}]])

        # Top row: Overall Similarity Graph
        fig.add_trace(
            go.Heatmap(
                x=similarity_correlation.columns,
                y=similarity_correlation.columns,
                z=similarity_correlation.round(2),
                coloraxis='coloraxis',
                xaxis='x',
                yaxis='y',
                hovertemplate=(
                    '<b>Column Pair</b><br>(%{x},%{y})<br><br>Similarity: '
                    '%{z}<extra></extra>'
                ),
            ),
            1,
            1,
        )

        # Real correlation heatmap
        fig.add_trace(
            go.Heatmap(
                x=real_correlation.columns,
                y=real_correlation.columns,
                z=real_correlation.round(2),
                coloraxis='coloraxis2',
                xaxis='x2',
                yaxis='y2',
                # Compare against synthetic data in the tooltip.
                customdata=synthetic_correlation.round(2),
                hovertemplate=(
                    '<b>Correlation</b><br>(%{x},%{y})<br><br>Real: %{z}'
                    '<br>(vs. Synthetic: %{customdata})<extra></extra>'
                ),
            ),
            2,
            1,
        )

        # Synthetic correlation heatmap
        fig.add_trace(
            go.Heatmap(
                x=synthetic_correlation.columns,
                y=synthetic_correlation.columns,
                z=synthetic_correlation.round(2),
                coloraxis='coloraxis2',
                xaxis='x3',
                yaxis='y3',
                # Compare against real data in the tooltip.
                customdata=real_correlation.round(2),
                hovertemplate=(
                    '<b>Correlation</b><br>(%{x},%{y})<br><br>Synthetic: '
                    '%{z}<br>(vs. Real: %{customdata})<extra></extra>'
                ),
            ),
            2,
            2,
        )

        fig.update_layout(
            title_text=f'Data Quality: Column Pair Trends (Average Score={round(average_score, 2)})',
            # Similarity heatmap color axis
            coloraxis={
                'colorbar_len': 0.5,
                'colorbar_x': 0.8,
                'colorbar_y': 0.8,
                'cmin': 0,
                'cmax': 1,
                'colorscale': ['#FF0000', '#F16141', '#36B37E'],
            },
            # Correlation heatmaps color axis
            coloraxis2={
                'colorbar_len': 0.5,
                'colorbar_y': 0.2,
                'cmin': -1,
                'cmax': 1,
                'colorscale': ['#03AFF1', '#000036', '#01E0C9'],
            },
            # Sync the zoom and pan of the bottom 2 graphs
            yaxis3={'visible': False, 'matches': 'y2'},
            xaxis3={'matches': 'x2'},
            height=900,
            width=900,
        )

        fig.update_yaxes(autorange='reversed')

        return fig

