import itertools
import warnings

import numpy as np
import pandas as pd
from plotly import graph_objects as go
from plotly.subplots import make_subplots

from sdmetrics.column_pairs.statistical.contingency_similarity import ContingencySimilarity
from sdmetrics.column_pairs.statistical.correlation_similarity import CorrelationSimilarity
from sdmetrics.reports.single_table._properties import BaseSingleTableProperty
from sdmetrics.utils import create_unique_name, is_datetime


class ColumnPairTrends(BaseSingleTableProperty):
    """Column pair trends property.

    This property evaluates the matching in trends between pairs of real
    and synthetic data columns. Each pair's correlation is calculated and
    The final score represents the average of these measures across all column pairs
    """

    _sdtype_to_shape = {
        'numerical': 'continuous',
        'datetime': 'continuous',
        'categorical': 'discrete',
        'boolean': 'discrete'
    }

    def __init__(self):
        self._warning_messages_list = []

    def _discretize_column(self, column_name, data, metadata, bin_edges=None):
        """Discretize a column.

        Args:
            column_name (str):
                The column name.
            data (pandas.Series):
                The column to discretize.
            metadta (dict):
                The table metadata.
            bin_edges (list):
                The bin edges to use for discretization.
                Defaults to None.

        Returns:
            pandas.Series:
                The discretized column.
        """
        column_result = data.copy()
        column_meta = metadata['columns'][column_name]
        col_sdtype = column_meta['sdtype']
        try:
            if col_sdtype == 'datetime' and not is_datetime(column_result):
                datetime_format = column_meta.get('format') or column_meta.get('datetime_format')
                column_result = pd.to_datetime(column_result, format=datetime_format)

            column_result = pd.to_numeric(column_result)
            if bin_edges is None:
                bin_edges = np.histogram_bin_edges(column_result.dropna())

            column_result = np.digitize(column_result, bins=bin_edges)
        except Exception as e:
            column_result = None
            message = (
                f"Unable to discretize '{column_name}'. No column pair trends metric "
                'will be calculated between this column and boolean/categorical columns.'
                f' Encountered Error: {type(e).__name__} {e}'
            )
            self._warning_messages_list.append(message)

        return column_result, bin_edges

    def _get_processed_data(self, real_data, synthetic_data, metadata):
        """Get the processed data for the property.

        Preprocess the data by converting datetime columns to numeric and
        adding a discrete version of datetime and numerical columns to the dataframes.

        Args:
            real_data (pandas.DataFrame):
                The real data
            synthetic_data (pandas.DataFrame):
                The synthetic data
            metadata (dict):
                The metadata of the table
        """
        processed_real_data = real_data.copy()
        processed_synthetic_data = synthetic_data.copy()

        for column_name in metadata['columns']:
            column_meta = metadata['columns'][column_name]
            column_sdtype = column_meta['sdtype']
            if column_sdtype in ['numerical', 'datetime']:
                name_discrete = create_unique_name(column_name + '_discrete', metadata['columns'])
                processed_real_data[name_discrete], bin_edges = self._discretize_column(
                    column_name, real_data[column_name], metadata
                )
                processed_synthetic_data[name_discrete], _ = self._discretize_column(
                    column_name, synthetic_data[column_name], metadata, bin_edges=bin_edges
                )
                is_datetime_sdtype = column_sdtype == 'datetime'
                if (is_datetime_sdtype and not is_datetime(processed_real_data[column_name])):
                    datetime_format = column_meta.get('format', column_meta.get('datetime_format'))
                    processed_real_data[column_name] = pd.to_datetime(
                        processed_real_data[column_name], format=datetime_format
                    )
                    processed_synthetic_data[column_name] = pd.to_datetime(
                        processed_synthetic_data[column_name], format=datetime_format
                    )

        return processed_real_data, processed_synthetic_data

    def _get_metric_and_columns(
            self, column_name_1, column_name_2, real_data, synthetic_data, metadata):
        """Get the metric and columns to use for the property.

        If one of the columns is discrete, use the ContingencySimilarity metric.
        Otherwise, use the CorrelationSimilarity metric.
        If ContingencySimilarity metric is used and one of the column is numerical or datetime,
        take the discrete version of the column.

        Args:
            column_name_1 (str):
                The name of the first column
            column_name_2 (str):
                The name of the second column
            real_data (pandas.DataFrame):
                The real data
            synthetic_data (pandas.DataFrame):
                The synthetic data
            metadata (dict):
                The metadata of the table
        """
        sdtype_col_1 = metadata['columns'][column_name_1]['sdtype']
        sdtype_col_2 = metadata['columns'][column_name_2]['sdtype']

        is_col_1_discrete = self._sdtype_to_shape[sdtype_col_1] == 'discrete'
        is_col_2_discrete = self._sdtype_to_shape[sdtype_col_2] == 'discrete'

        col_name_1 = column_name_1
        col_name_2 = column_name_2
        if is_col_1_discrete or is_col_2_discrete:
            metric = ContingencySimilarity
            if sdtype_col_1 in ['datetime', 'numerical']:
                col_name_1 = create_unique_name(column_name_1 + '_discrete', metadata['columns'])
            if sdtype_col_2 in ['datetime', 'numerical']:
                col_name_2 = create_unique_name(column_name_2 + '_discrete', metadata['columns'])
        else:
            metric = CorrelationSimilarity

        columns_real = real_data[[col_name_1, col_name_2]]
        columns_synthetic = synthetic_data[[col_name_1, col_name_2]]

        return metric, columns_real, columns_synthetic

    def _safe_get_metric_and_columns(
            self, column_name_1, column_name_2, real_data, synthetic_data, metadata):
        """Call ``_get_metric_and_columns`` and catch any exceptions.

        Args:
            column_name_1 (str):
                The name of the first column
            column_name_2 (str):
                The name of the second column
            real_data (pandas.DataFrame):
                The real data
            synthetic_data (pandas.DataFrame):
                The synthetic data
            metadata (dict):
                The metadata of the table
        """
        try:
            return self._get_metric_and_columns(
                column_name_1, column_name_2, real_data, synthetic_data, metadata
            )
        except Exception:
            return None, None, None

    def _safe_compute_breakdown(
            self, column_name_1, column_name_2, metric, columns_real, columns_synthetic):
        """Call ``metric.compute_breakdown`` and catch any exceptions.

        Args:
            column_name_1 (str):
                The name of the first column
            column_name_2 (str):
                The name of the second column
            metric (Metric):
                The metric to use
            columns_real (pandas.DataFrame):
                The real data
            columns_synthetic (pandas.DataFrame):
                The synthetic data
        """
        try:
            return metric.compute_breakdown(
                real_data=columns_real, synthetic_data=columns_synthetic
            )
        except Exception as e:
            message = (
                f"Unable to compute Column Pair Trends for column ('{column_name_1}', "
                f"'{column_name_2}'). Encountered Error: {type(e).__name__} {e}"
            )
            self._warning_messages_list.append(message)
            return {'score': np.nan, 'real': np.nan, 'synthetic': np.nan}

    def _handling_warning_messages(self):
        """Handle warning messages.

        Remove duplicate messages and raise the warnings.
        """
        if self._warning_messages_list:
            self._warning_messages_list = list(dict.fromkeys(self._warning_messages_list).keys())
            for message in self._warning_messages_list:
                warnings.warn(message)

    def _generate_details(self, real_data, synthetic_data, metadata, progress_bar):
        """Generate the _details dataframe for the column pair trends property.

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
        processed_real_data, processed_synthetic_data = self._get_processed_data(
            real_data, synthetic_data, metadata
        )

        column_names_1 = []
        column_names_2 = []
        metric_names = []
        scores = []
        real_correlations = []
        synthetic_correaltions = []

        list_col_names = list(metadata['columns'])
        list_dtypes = self._sdtype_to_shape.keys()
        for column_names in itertools.combinations(list_col_names, r=2):
            column_name_1 = column_names[0]
            column_name_2 = column_names[1]

            sdtype_col_1 = metadata['columns'][column_name_1]['sdtype']
            sdtype_col_2 = metadata['columns'][column_name_2]['sdtype']

            valid_sdtypes = sdtype_col_1 in list_dtypes and sdtype_col_2 in list_dtypes
            if valid_sdtypes:
                metric, columns_real, columns_synthetic = self._safe_get_metric_and_columns(
                    column_name_1, column_name_2, processed_real_data,
                    processed_synthetic_data, metadata
                )
                if metric is None:
                    pair_score = np.nan
                    real_correlation = np.nan
                    synthetic_correlation = np.nan
                else:
                    score_breakdown = self._safe_compute_breakdown(
                        column_name_1, column_name_2, metric, columns_real, columns_synthetic)
                    pair_score = score_breakdown['score']
                    if metric.__name__ == 'CorrelationSimilarity':
                        real_correlation = score_breakdown['real']
                        synthetic_correlation = score_breakdown['synthetic']
                    else:
                        real_correlation = np.nan
                        synthetic_correlation = np.nan

                column_names_1.append(column_name_1)
                column_names_2.append(column_name_2)
                metric_names.append(metric.__name__)
                scores.append(pair_score)
                real_correlations.append(real_correlation)
                synthetic_correaltions.append(synthetic_correlation)

            if progress_bar:
                progress_bar.update()

        if progress_bar:
            progress_bar.close()

        self._handling_warning_messages()
        result = pd.DataFrame({
            'Column 1': column_names_1,
            'Column 2': column_names_2,
            'Metric': metric_names,
            'Score': scores,
            'Real Correlation': real_correlations,
            'Synthetic Correlation': synthetic_correaltions
        })

        return result

    def _get_correlation_matrix(self, column_name):
        """Get the correlation matrix for the given column name."""
        if column_name not in ['Score', 'Real Correlation', 'Synthetic Correlation']:
            raise ValueError(f"Invalid column name for _get_correlation_matrix : '{column_name}'")

        table = self._details.dropna(subset=[column_name])
        names = list(pd.concat([table['Column 1'], table['Column 2']]).unique())
        heatmap_df = pd.DataFrame(index=names, columns=names)

        for idx_1, column_name_1 in enumerate(names):
            for column_name_2 in names[idx_1:]:
                if column_name_1 == column_name_2:
                    heatmap_df.loc[column_name_1, column_name_2] = 1
                    continue

                col_1_cond = (table['Column 1'] == column_name_1)
                col_2_cond = (table['Column 2'] == column_name_2)
                if table.loc[col_1_cond & col_2_cond].empty:
                    col_1_cond = (table['Column 1'] == column_name_2)
                    col_2_cond = (table['Column 2'] == column_name_1)

                if not table.loc[col_1_cond & col_2_cond].empty:
                    score = table.loc[col_1_cond & col_2_cond][column_name].array[0]
                    heatmap_df.loc[column_name_1, column_name_2] = score
                    heatmap_df.loc[column_name_2, column_name_1] = score

        heatmap_df = heatmap_df.astype(float)

        return heatmap_df.round(3)

    def _get_heatmap(self, correlation_matrix, coloraxis, hovertemplate, customdata=None):
        """Get the heatmap for the given correlation matrix."""
        fig = go.Heatmap(
                x=correlation_matrix.columns,
                y=correlation_matrix.columns,
                z=correlation_matrix,
                coloraxis=coloraxis,
                customdata=customdata,
                hovertemplate=hovertemplate,
            )

        return fig

    def _update_layout(self, fig):
        """Update the layout of the figure."""
        average_score = self._compute_average()
        color_dict = {
            'colorbar_len': 0.5,
            'cmin': 0,
            'cmax': 1,
        }

        colors_1 = ['#FF0000', '#F16141', '#36B37E']
        colors_2 = ['#03AFF1', '#000036', '#01E0C9']

        fig.update_layout(
            title_text=f'Data Quality: Column Pair Trends (Average Score={average_score})',
            coloraxis={**color_dict, 'colorbar_x': 0.8, 'colorbar_y': 0.8, 'colorscale': colors_1},
            coloraxis2={**color_dict, 'colorbar_y': 0.2, 'cmin': -1, 'colorscale': colors_2},
            yaxis3={'visible': False, 'matches': 'y2'},
            xaxis3={'matches': 'x2'},
            height=900,
            width=900,
        )

        fig.update_yaxes(autorange='reversed')

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
        similarity_correlation = self._get_correlation_matrix('Score')
        real_correlation = self._get_correlation_matrix('Real Correlation')
        synthetic_correlation = self._get_correlation_matrix('Synthetic Correlation')

        titles = [
                'Real vs. Synthetic Similarity',
                'Numerical Correlation (Real Data)',
                'Numerical Correlation (Synthetic Data)',
        ]
        specs = [[{'colspan': 2, 'l': 0.26, 'r': 0.26}, None], [{}, {}]]
        tmpl_1 = '<b>Column Pair</b><br>(%{x},%{y})<br><br>Similarity: %{z}<extra></extra>'
        tmpl_2 = (
            '<b>Correlation</b><br>(%{x},%{y})<br><br>Synthetic: %{z}<br>(vs. Real: '
            '%{customdata})<extra></extra>'
        )

        fig = make_subplots(rows=2, cols=2, subplot_titles=titles, specs=specs)

        fig.add_trace(
            self._get_heatmap(similarity_correlation, 'coloraxis', tmpl_1), 1, 1
        )
        fig.add_trace(
            self._get_heatmap(real_correlation, 'coloraxis2', tmpl_2, synthetic_correlation), 2, 1
        )
        fig.add_trace(
            self._get_heatmap(synthetic_correlation, 'coloraxis2', tmpl_2, real_correlation), 2, 2
        )

        self._update_layout(fig)

        return fig
