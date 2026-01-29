import itertools

import numpy as np
import pandas as pd
from plotly import graph_objects as go
from plotly.subplots import make_subplots

from sdmetrics._utils_metadata import _convert_datetime_column
from sdmetrics.column_pairs.statistical import ContingencySimilarity, CorrelationSimilarity
from sdmetrics.reports.single_table._properties import BaseSingleTableProperty
from sdmetrics.reports.utils import PlotConfig


class ColumnPairTrends(BaseSingleTableProperty):
    """Column pair trends property.

    This property evaluates the matching in trends between pairs of real
    and synthetic data columns. Each pair's correlation is calculated and
    the final score represents the average of these measures across all column pairs
    """

    _num_iteration_case = 'column_pair'
    _sdtype_to_shape = {
        'numerical': 'continuous',
        'datetime': 'continuous',
        'categorical': 'discrete',
        'boolean': 'discrete',
    }

    def __init__(self):
        super().__init__()
        self._columns_datetime_conversion_failed = {}
        self._columns_discretization_failed = {}
        self.real_correlation_threshold = 0
        self.real_association_threshold = 0

    def _compute_average(self):
        """Average the scores for each column pair, honoring contribution rules."""
        return self._compute_average_with_threshold('Meets Threshold?')

    def _convert_datetime_columns_to_numeric(self, data, metadata):
        """Convert all the datetime columns to numeric columns.

        Args:
            data (pandas.DataFrame):
                The data to convert.
            metadata (dict):
                The table metadata.

        Returns:
            pandas.Series:
                The converted column.
        """
        for column_name in metadata['columns']:
            column_meta = metadata['columns'][column_name]
            col_sdtype = column_meta['sdtype']
            try:
                if col_sdtype == 'datetime':
                    data[column_name] = _convert_datetime_column(
                        column_name, data[column_name], column_meta
                    )
                    nan_mask = pd.isna(data[column_name])
                    data[column_name] = pd.to_numeric(data[column_name])
                    if nan_mask.any():
                        data.loc[nan_mask, column_name] = np.nan

                continue
            except Exception as e:
                message = f'{type(e).__name__}: {e}'
                self._columns_datetime_conversion_failed[column_name] = message
                continue

        return data

    def _discretize_column(self, column_name, data, bin_edges=None):
        """Discretize a column.

        Args:
            column_name (str):
                The name of the column to discretize.
            data (pandas.Series):
                The column to discretize.
            bin_edges (list):
                The bin edges to use for discretization.
                Defaults to None.

        Returns:
            pandas.Series:
                The discretized column.
        """
        column_result = data.copy()
        try:
            if bin_edges is None:
                bin_edges = np.histogram_bin_edges(column_result.dropna())

            column_result = np.digitize(column_result, bins=bin_edges)

        except Exception as e:
            message = f'{type(e).__name__}: {e}'
            self._columns_discretization_failed[column_name] = message

        return column_result, bin_edges

    def _get_processed_data(self, data, metadata):
        """Get the processed data for the property.

        Preprocess the data by converting datetime columns to numeric and
        adding a discrete version of datetime and numerical columns to the dataframes.

        Args:
            data (pandas.DataFrame):
                The data
            metadata (dict):
                The metadata of the table
        """
        processed_data = data.copy()
        discretized_dict = {}

        processed_data = self._convert_datetime_columns_to_numeric(processed_data, metadata)

        for column_name in metadata['columns']:
            column_meta = metadata['columns'][column_name]
            column_sdtype = column_meta['sdtype']
            if column_sdtype in ['numerical', 'datetime']:
                discretized_dict[column_name], _ = self._discretize_column(
                    column_name, processed_data[column_name]
                )

        return processed_data, pd.DataFrame(discretized_dict, index=processed_data.index)

    def _get_columns_data_and_metric(
        self,
        column_name_1,
        column_name_2,
        real_data,
        real_discrete_data,
        synthetic_data,
        synthetic_discrete_data,
        metadata,
    ):
        """Get the data and the metric for the property.

        If one is comparing a continuous column to a discrete column, use the discrete version
        of the continuous column and the Contingency metric. Otherwise use the original columns.
        If the columns are both continuous, use the Correlation metric. If the columns are both
        discrete, use the Contingency metric.

        Args:
            column_name_1 (str):
                The name of the first column
            column_name_2 (str):
                The name of the second column
            real_data (pandas.DataFrame):
                The real data
            real_discrete_data (pandas.DataFrame):
                The real data with discrete versions of the continuous columns
            synthetic_data (pandas.DataFrame):
                The synthetic data
            synthetic_discrete_data (pandas.DataFrame):
                The synthetic data with discrete versions of the continuous columns
            metadata (dict):
                The metadata of the table
        """
        sdtype_col_1 = metadata['columns'][column_name_1]['sdtype']
        sdtype_col_2 = metadata['columns'][column_name_2]['sdtype']
        if self._sdtype_to_shape[sdtype_col_1] != self._sdtype_to_shape[sdtype_col_2]:
            metric = ContingencySimilarity
            if self._sdtype_to_shape[sdtype_col_1] == 'continuous':
                real_col_1 = real_discrete_data[column_name_1]
                synthetic_col_1 = synthetic_discrete_data[column_name_1]
                real_col_2 = real_data[column_name_2]
                synthetic_col_2 = synthetic_data[column_name_2]
            else:
                real_col_1 = real_data[column_name_1]
                synthetic_col_1 = synthetic_data[column_name_1]
                real_col_2 = real_discrete_data[column_name_2]
                synthetic_col_2 = synthetic_discrete_data[column_name_2]

            data_real = pd.concat([real_col_1, real_col_2], axis=1)
            data_synthetic = pd.concat([synthetic_col_1, synthetic_col_2], axis=1)
        else:
            if self._sdtype_to_shape[sdtype_col_1] == 'continuous':
                metric = CorrelationSimilarity
            else:
                metric = ContingencySimilarity

            data_real = real_data[[column_name_1, column_name_2]]
            data_synthetic = synthetic_data[[column_name_1, column_name_2]]

        return data_real, data_synthetic, metric

    def _preprocessing_failed(self, column_name_1, column_name_2, sdtype_col_1, sdtype_col_2):
        """Check if a processing of one of the columns has failed.

        Args:
            column_name_1 (str):
                The name of the first column
            column_name_2 (str):
                The name of the second column
            sdtype_col_1 (str):
                The sdtype of the first column
            sdtype_col_2 (str):
                The sdtype of the second column
        """
        error = None
        if column_name_1 in self._columns_datetime_conversion_failed.keys():
            error = self._columns_datetime_conversion_failed[column_name_1]

        elif column_name_2 in self._columns_datetime_conversion_failed.keys():
            error = self._columns_datetime_conversion_failed[column_name_2]

        elif self._sdtype_to_shape[sdtype_col_1] != self._sdtype_to_shape[sdtype_col_2]:
            if column_name_1 in self._columns_discretization_failed.keys():
                error = self._columns_discretization_failed[column_name_1]
            elif column_name_2 in self._columns_discretization_failed.keys():
                error = self._columns_discretization_failed[column_name_2]

        return error

    def _compute_pair_score(
        self,
        metric,
        col_real,
        col_synthetic,
        metric_params,
    ):
        """Compute the score breakdown and threshold check for a column pair."""
        params = dict(metric_params)
        if metric.__name__ == 'CorrelationSimilarity':
            if self.real_correlation_threshold > 0:
                params['real_correlation_threshold'] = self.real_correlation_threshold

            score_breakdown = metric.compute_breakdown(
                real_data=col_real, synthetic_data=col_synthetic, **params
            )
            pair_score = score_breakdown['score']
            real_correlation = score_breakdown['real']
            synthetic_correlation = score_breakdown['synthetic']
            real_association = np.nan
            if self.real_correlation_threshold <= 0:
                meets_threshold_pair = True
            else:
                meets_threshold_pair = (
                    not np.isnan(real_correlation)
                    and abs(real_correlation) > self.real_correlation_threshold
                )
        else:
            if self.real_association_threshold > 0:
                params['real_association_threshold'] = self.real_association_threshold

            score_breakdown = metric.compute_breakdown(
                real_data=col_real, synthetic_data=col_synthetic, **params
            )
            pair_score = score_breakdown['score']
            real_correlation = np.nan
            synthetic_correlation = np.nan
            real_association = score_breakdown.get('real_association', np.nan)
            if self.real_association_threshold > 0:
                meets_threshold_pair = (
                    not np.isnan(real_association)
                    and real_association > self.real_association_threshold
                )
            else:
                meets_threshold_pair = True

        return (
            pair_score,
            real_correlation,
            synthetic_correlation,
            real_association,
            meets_threshold_pair,
        )

    def _generate_details(
        self, real_data, synthetic_data, metadata, progress_bar=None, column_pairs=None
    ):
        """Generate the _details dataframe for the column pair trends property.

        Args:
            real_data (pandas.DataFrame):
                The real data
            synthetic_data (pandas.DataFrame):
                The synthetic data
            metadata (dict):
                The metadata of the table
            progress_bar:
                The progress bar to use. Defaults to None.
            column_pairs (list[tuple[str, str]]):
                Pairs of columns to calculate results for. If None, uses every combination of
                pairs of columns in the metadata. Defaults to None.
        """
        processed_real_data, discrete_real = self._get_processed_data(real_data, metadata)
        processed_synthetic_data, discrete_synthetic = self._get_processed_data(
            synthetic_data, metadata
        )

        column_names_1 = []
        column_names_2 = []
        metric_names = []
        scores = []
        real_correlations = []
        synthetic_correlations = []
        real_associations = []
        meets_threshold = []
        error_messages = []

        list_dtypes = self._sdtype_to_shape.keys()

        column_pairs = (
            itertools.combinations(list(metadata['columns']), r=2)
            if column_pairs is None
            else column_pairs
        )
        for column_names in column_pairs:
            column_name_1 = column_names[0]
            column_name_2 = column_names[1]

            sdtype_col_1 = metadata['columns'][column_name_1]['sdtype']
            sdtype_col_2 = metadata['columns'][column_name_2]['sdtype']

            error = None
            valid_sdtypes = sdtype_col_1 in list_dtypes and sdtype_col_2 in list_dtypes
            if not valid_sdtypes:
                if progress_bar:
                    progress_bar.update()

                continue

            col_real, col_synthetic, metric = self._get_columns_data_and_metric(
                column_name_1,
                column_name_2,
                processed_real_data,
                discrete_real,
                processed_synthetic_data,
                discrete_synthetic,
                metadata,
            )

            metric_params = {}
            if (
                self.num_rows_subsample
                and (metric == ContingencySimilarity)
                and (max(len(col_real), len(col_synthetic)) > self.num_rows_subsample)
            ):
                metric_params['num_rows_subsample'] = self.num_rows_subsample

            try:
                error = self._preprocessing_failed(
                    column_name_1, column_name_2, sdtype_col_1, sdtype_col_2
                )
                if error:
                    raise Exception('Preprocessing failed')

                (
                    pair_score,
                    real_correlation,
                    synthetic_correlation,
                    real_association,
                    meets_threshold_pair,
                ) = self._compute_pair_score(metric, col_real, col_synthetic, metric_params)

            except Exception as e:
                pair_score = np.nan
                real_correlation = np.nan
                synthetic_correlation = np.nan
                real_association = np.nan
                meets_threshold_pair = np.nan
                if not str(e) == 'Preprocessing failed':
                    error = f'{type(e).__name__}: {e}'
            finally:
                if progress_bar:
                    progress_bar.update()

            column_names_1.append(column_name_1)
            column_names_2.append(column_name_2)
            metric_names.append(metric.__name__)
            scores.append(pair_score)
            real_correlations.append(real_correlation)
            synthetic_correlations.append(synthetic_correlation)
            real_associations.append(real_association)
            meets_threshold.append(meets_threshold_pair)
            error_messages.append(error)

        result = pd.DataFrame({
            'Column 1': column_names_1,
            'Column 2': column_names_2,
            'Metric': metric_names,
            'Score': scores,
            'Real Correlation': real_correlations,
            'Synthetic Correlation': synthetic_correlations,
            'Real Association': real_associations,
            'Meets Threshold?': meets_threshold,
            'Error': error_messages,
        })

        if result['Error'].isna().all():
            result = result.drop('Error', axis=1)

        return result

    def _get_correlation_table(self, column_name):
        if column_name not in ['Score', 'Real Correlation', 'Synthetic Correlation']:
            raise ValueError(f"Invalid column name for _get_correlation_matrix : '{column_name}'")

        table = self.details.copy()
        if 'Error' in table.columns:
            table = table[table['Error'].isna()]

        if column_name in ['Real Correlation', 'Synthetic Correlation']:
            table = table[table['Metric'] == 'CorrelationSimilarity']

        return table.dropna(subset=['Score'])

    def _get_pair_score(self, table, column_name_1, column_name_2, column_name):
        # check whether the combination (Column 1, Column 2) or (Column 2, Column 1) is in the table
        col_1_loc = table['Column 1'] == column_name_1
        col_2_loc = table['Column 2'] == column_name_2
        if table.loc[col_1_loc & col_2_loc].empty:
            col_1_loc = table['Column 1'] == column_name_2
            col_2_loc = table['Column 2'] == column_name_1

        if not table.loc[col_1_loc & col_2_loc].empty:
            return table.loc[col_1_loc & col_2_loc][column_name].array[0]

        return None

    def _get_correlation_matrix(self, column_name):
        """Get the correlation matrix for the given column name.

        Args:
            column_name (str):
                The column name to use.
        """
        table = self._get_correlation_table(column_name)
        if table.empty:
            return pd.DataFrame()

        names = list(pd.concat([table['Column 1'], table['Column 2']]).unique())
        available_columns = set(names)
        heatmap_df = pd.DataFrame(index=names, columns=names)

        for idx_1, column_name_1 in enumerate(names):
            for column_name_2 in names[idx_1:]:
                if column_name_1 == column_name_2:
                    heatmap_df.loc[column_name_1, column_name_2] = (
                        1 if column_name_1 in available_columns else np.nan
                    )
                    continue

                score = self._get_pair_score(table, column_name_1, column_name_2, column_name)
                if score is not None:
                    heatmap_df.loc[column_name_1, column_name_2] = score
                    heatmap_df.loc[column_name_2, column_name_1] = score

        return heatmap_df.astype(float).round(3)

    def _get_heatmap(self, correlation_matrix, coloraxis, hovertemplate, customdata=None):
        """Get the heatmap for the given correlation matrix.

        Args:
            correlation_matrix (pandas.DataFrame):
                The correlation matrix to use.
            coloraxis (str):
                The coloraxis to use.
            hovertemplate (str):
                The hovertemplate to use.
            customdata (pandas.DataFrame or None):
                The customdata to use. Defaults to None.
        """
        base_heatmap = go.Heatmap(
            x=correlation_matrix.columns,
            y=correlation_matrix.columns,
            z=correlation_matrix,
            coloraxis=coloraxis,
            customdata=customdata,
            hovertemplate=hovertemplate,
        )

        nan_mask = correlation_matrix.isna().to_numpy()
        if not nan_mask.any():
            return [base_heatmap]

        nan_heatmap = go.Heatmap(
            x=correlation_matrix.columns,
            y=correlation_matrix.columns,
            z=np.where(nan_mask, 1, np.nan),
            colorscale=[[0, '#B0B0B0'], [1, '#B0B0B0']],
            showscale=False,
            hoverinfo='skip',
        )

        return [base_heatmap, nan_heatmap]

    def _update_layout(self, fig, show_correlations=True):
        """Update the layout of the figure.

        Args:
            fig (plotly.graph_objects._figure.Figure):
                The figure to update.
            show_correlations (bool):
                Whether or not to show numerical correlation plots.
        """
        average_score = round(self._compute_average(), 2)
        color_dict = {
            'colorbar_len': 0.5,
            'cmin': 0,
            'cmax': 1,
        }

        colors_1 = [PlotConfig.RED, PlotConfig.ORANGE, PlotConfig.GREEN]
        colors_2 = [PlotConfig.DATACEBO_BLUE, PlotConfig.DATACEBO_DARK, PlotConfig.DATACEBO_GREEN]

        layout_args = {
            'title_text': (f'Data Quality: Column Pair Trends (Average Score={average_score})'),
            'coloraxis': {
                **color_dict,
                'colorbar_x': 0.8,
                'colorbar_y': 0.8,
                'colorscale': colors_1,
            },
            'height': 900 if show_correlations else 450,
            'width': 900,
            'font': {'size': PlotConfig.FONT_SIZE},
        }

        if show_correlations:
            layout_args.update({
                'coloraxis2': {**color_dict, 'colorbar_y': 0.2, 'cmin': -1, 'colorscale': colors_2},
                'yaxis3': {'visible': False, 'matches': 'y2'},
                'xaxis3': {'matches': 'x2'},
            })
        else:
            layout_args['coloraxis'].pop('colorbar_len', None)
            layout_args['coloraxis']['colorbar'] = {
                'x': 0.92,
                'xanchor': 'left',
                'y': 0.5,
                'yanchor': 'middle',
                'len': 1,
            }
            layout_args['xaxis'] = {'constrain': 'domain', 'domain': [0.1, 0.9]}
            layout_args['yaxis'] = {'scaleanchor': 'x', 'scaleratio': 1}
            layout_args['margin'] = {'r': 60}

        fig.update_layout(**layout_args)
        fig.update_yaxes(autorange='reversed')

    def _get_empty_visualization(self):
        fig = go.Figure()
        fig.update_layout(
            title_text='No data to plot',
            xaxis={'visible': False},
            yaxis={'visible': False},
            showlegend=False,
            plot_bgcolor='rgba(0,0,0,0)',
            paper_bgcolor='white',
        )

        return fig

    def _add_heatmap_traces(
        self,
        fig,
        correlation_matrix,
        coloraxis,
        hovertemplate,
        customdata=None,
        row=None,
        col=None,
    ):
        for trace in self._get_heatmap(
            correlation_matrix, coloraxis, hovertemplate, customdata=customdata
        ):
            if row is None or col is None:
                fig.add_trace(trace)
            else:
                fig.add_trace(trace, row, col)

    def _build_single_heatmap_figure(self, similarity_correlation, title):
        fig = make_subplots(rows=1, cols=1, subplot_titles=[title])
        fig.update_xaxes(tickangle=45)
        tmpl_1 = '<b>Column Pair</b><br>(%{x},%{y})<br><br>Similarity: %{z}<extra></extra>'
        self._add_heatmap_traces(fig, similarity_correlation, 'coloraxis', tmpl_1)
        self._update_layout(fig, show_correlations=False)

        return fig

    def _build_full_heatmap_figure(
        self,
        similarity_correlation,
        real_correlation,
        synthetic_correlation,
        titles,
    ):
        tmpl_1 = '<b>Column Pair</b><br>(%{x},%{y})<br><br>Similarity: %{z}<extra></extra>'
        tmpl_2 = (
            '<b>Correlation</b><br>(%{x},%{y})<br><br>Synthetic: %{z}<br>(vs. Real: '
            '%{customdata})<extra></extra>'
        )
        specs = [[{'colspan': 2, 'l': 0.26, 'r': 0.26}, None], [{}, {}]]
        fig = make_subplots(rows=2, cols=2, subplot_titles=titles, specs=specs)
        fig.update_xaxes(tickangle=45)
        self._add_heatmap_traces(fig, similarity_correlation, 'coloraxis', tmpl_1)

        synthetic_correlation = synthetic_correlation.reindex(
            index=real_correlation.index, columns=real_correlation.columns
        )
        self._add_heatmap_traces(
            fig,
            real_correlation,
            'coloraxis2',
            tmpl_2,
            customdata=synthetic_correlation,
            row=2,
            col=1,
        )
        self._add_heatmap_traces(
            fig,
            synthetic_correlation,
            'coloraxis2',
            tmpl_2,
            customdata=real_correlation,
            row=2,
            col=2,
        )

        self._update_layout(fig)

        return fig

    def get_visualization(self):
        """Create a plot to show the column pairs data.

        This plot will have one graph in the top row and two in the bottom row.

        Returns:
            plotly.graph_objects._figure.Figure
        """
        similarity_correlation = self._get_correlation_matrix('Score')
        if similarity_correlation.empty:
            return self._get_empty_visualization()

        real_correlation = self._get_correlation_matrix('Real Correlation')
        synthetic_correlation = self._get_correlation_matrix('Synthetic Correlation')

        titles = [
            'Real vs. Synthetic Similarity',
            'Numerical Correlation (Real Data)',
            'Numerical Correlation (Synthetic Data)',
        ]
        if real_correlation.empty:
            return self._build_single_heatmap_figure(similarity_correlation, titles[0])

        return self._build_full_heatmap_figure(
            similarity_correlation,
            real_correlation,
            synthetic_correlation,
            titles,
        )
