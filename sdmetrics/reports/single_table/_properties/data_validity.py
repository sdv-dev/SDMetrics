import numpy as np
import pandas as pd
import plotly.express as px

from sdmetrics._utils_metadata import _convert_column_to_string
from sdmetrics.reports.single_table._properties import BaseSingleTableProperty
from sdmetrics.reports.utils import PlotConfig
from sdmetrics.single_column import (
    BoundaryAdherence,
    CategoryAdherence,
    DatetimeFormatAdherence,
    KeyUniqueness,
    RegexFormatAdherence,
)
from sdmetrics.utils import (
    get_alternate_keys,
    get_columns_from_metadata,
    get_primary_key_from_metadata,
    get_sequence_index,
)


class DataValidity(BaseSingleTableProperty):
    """Data Validity property class for single table.

    This property computes, at base, whether each column contains valid data.
    The metric is based on the type data in each column.
    The BoundaryAdherence metric is used for numerical and datetime columns, the CategoryAdherence
    is used for categorical, ordinal and boolean columns and the KeyUniqueness for primary
    and alternate keys. The other column types are ignored by this property.

    If the metadata defines the valid range of a column, it is passed down to the metric,
    which uses it instead of computing it from the real data.
    """

    _num_iteration_case = 'column'
    _metric_to_arguments = {
        DatetimeFormatAdherence: ('datetime_format',),
        RegexFormatAdherence: ('regex_format',),
        BoundaryAdherence: ('range_min', 'range_max', 'range_is_nullable'),
        CategoryAdherence: ('range_values', 'range_is_nullable'),
    }
    _metric_to_required_argument = {
        DatetimeFormatAdherence: 'datetime_format',
        RegexFormatAdherence: 'regex_format',
    }
    _sdtype_to_metric = {
        'numerical': [BoundaryAdherence],
        'datetime': [BoundaryAdherence, DatetimeFormatAdherence],
        'categorical': [CategoryAdherence],
        'ordinal': [CategoryAdherence],
        'boolean': [CategoryAdherence],
        'id': [KeyUniqueness, RegexFormatAdherence],
    }

    @classmethod
    def _get_metric_arguments(cls, metric, column_name, columns_meta):
        """Get the information of a column defined in the metadata.

        Args:
            metric (SingleColumnMetric):
                The metric to compute the column score with.
            column_name (str):
                The name of the column.
            columns_meta (dict):
                The metadata of every column of the table.

        Returns:
            dict:
                The arguments to pass down to the metric. Any information that is not
                defined in the metadata is omitted.
        """
        arguments = cls._metric_to_arguments.get(metric)
        if not arguments:
            return {}

        column_meta = columns_meta[column_name]
        column_arguments = {
            argument: column_meta[argument] for argument in arguments if argument in column_meta
        }
        return column_arguments

    @classmethod
    def _get_column_metrics(cls, sdtype, column_name, columns_meta, is_unique):
        """Get the metrics that apply to a column.

        ``KeyUniqueness`` only applies to primary and alternate keys, and the format metrics
        only apply to the columns that define their format in the metadata.

        Args:
            sdtype (str or None):
                The sdtype of the column, or ``None``.
            column_name (str or list):
                The name of the column.
            columns_meta (dict):
                The metadata of every column of the table.
            is_unique (bool):
                Whether the column is a primary or an alternate key.

        Returns:
            list:
                The metrics to compute the column scores with.
        """

        def has_required_argument(metric):
            required_argument = cls._metric_to_required_argument.get(metric)
            return required_argument is None or required_argument in columns_meta[column_name]

        return [
            metric
            for metric in cls._sdtype_to_metric.get(sdtype, [KeyUniqueness])
            if (is_unique or metric is not KeyUniqueness) and has_required_argument(metric)
        ]

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
        columns_meta = get_columns_from_metadata(metadata)
        column_names, metric_names, scores = [], [], []
        column_sdtypes = [
            (col_name, col_meta['sdtype']) for col_name, col_meta in columns_meta.items()
        ]
        error_messages = []
        primary_key = get_primary_key_from_metadata(metadata)
        if isinstance(primary_key, list):
            if len(primary_key) > 1:
                column_sdtypes = [(primary_key, None)] + column_sdtypes
            else:
                primary_key = primary_key[0]

        alternate_keys = get_alternate_keys(metadata)
        sequence_index = get_sequence_index(metadata)

        for column_name, sdtype in column_sdtypes:
            primary_key_match = column_name == primary_key
            alternate_key_match = column_name in alternate_keys
            is_unique = primary_key_match or alternate_key_match
            is_sequence_index = column_name == sequence_index

            metrics = self._get_column_metrics(sdtype, column_name, columns_meta, is_unique)
            if is_sequence_index and BoundaryAdherence in metrics:
                metrics = []

            for metric in metrics:
                try:
                    metric_arguments = self._get_metric_arguments(metric, column_name, columns_meta)
                    if metric in [DatetimeFormatAdherence, RegexFormatAdherence]:
                        column_meta = columns_meta[column_name]
                        real_column = _convert_column_to_string(real_data[column_name], column_meta)
                        synthetic_column = _convert_column_to_string(
                            synthetic_data[column_name], column_meta
                        )
                        column_score = metric.compute(
                            real_column, synthetic_column, **metric_arguments
                        )
                    else:
                        column_score = metric.compute(
                            real_data[column_name],
                            synthetic_data[column_name],
                            **metric_arguments,
                        )

                    error_message = None

                except Exception as e:
                    column_score = np.nan
                    error_message = f'{type(e).__name__}: {e}'

                column_names.append(column_name)
                metric_names.append(metric.__name__)
                scores.append(column_score)
                error_messages.append(error_message)

            if progress_bar:
                progress_bar.update()

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
