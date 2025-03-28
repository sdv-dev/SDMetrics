"""Base Time Series metric class."""

from operator import attrgetter

from sdmetrics._utils_metadata import _convert_datetime_column, _validate_metadata_dict
from sdmetrics.base import BaseMetric
from sdmetrics.utils import get_columns_from_metadata


class TimeSeriesMetric(BaseMetric):
    """Base class for metrics that apply to time series.

    Attributes:
        name (str):
            Name to use when reports about this metric are printed.
        goal (sdmetrics.goal.Goal):
            The goal of this metric.
        min_value (Union[float, tuple[float]]):
            Minimum value or values that this metric can take.
        max_value (Union[float, tuple[float]]):
            Maximum value or values that this metric can take.
    """

    name = None
    goal = None
    min_value = None
    max_value = None

    _DTYPES_TO_TYPES = {
        'i': {
            'sdtype': 'numerical',
        },
        'f': {
            'sdtype': 'numerical',
        },
        'O': {
            'sdtype': 'categorical',
        },
        'b': {
            'sdtype': 'boolean',
        },
        'M': {
            'sdtype': 'datetime',
        },
    }

    @classmethod
    def _validate_inputs(cls, real_data, synthetic_data, metadata=None, sequence_key=None):
        if set(real_data.columns) != set(synthetic_data.columns):
            raise ValueError('`real_data` and `synthetic_data` must have the same columns')

        if metadata is not None:
            _validate_metadata_dict(metadata)
            fields = get_columns_from_metadata(metadata)
            for column in real_data.columns:
                if column not in fields:
                    raise ValueError(f'Column {column} not found in metadata')

            for field in fields.keys():
                if field not in real_data.columns:
                    raise ValueError(f'Field {field} not found in data')

            for column, col_metadata in metadata['columns'].items():
                if col_metadata['sdtype'] == 'datetime':
                    real_data[column] = _convert_datetime_column(
                        column, real_data[column], col_metadata
                    )
                    synthetic_data[column] = _convert_datetime_column(
                        column, synthetic_data[column], col_metadata
                    )

        else:
            dtype_kinds = real_data.dtypes.apply(attrgetter('kind'))
            metadata = {'columns': dtype_kinds.apply(cls._DTYPES_TO_TYPES.get).to_dict()}

        sequence_key = metadata.get('sequence_key', sequence_key or [])
        sequence_key = [sequence_key] if not isinstance(sequence_key, list) else sequence_key

        return metadata, sequence_key

    @classmethod
    def compute(cls, real_data, synthetic_data, metadata=None, sequence_key=None):
        """Compute this metric.

        Args:
            real_data (pandas.DataFrame):
                The values from the real dataset, passed as a pandas.DataFrame.
            synthetic_data (pandas.DataFrame):
                The values from the synthetic dataset, passed as a pandas.DataFrame.
            metadata (dict):
                TimeSeries metadata dict. If not passed, it is build based on the
                real_data fields and dtypes.
            sequence_key (list[str]):
                Names of the columns which identify different time series
                sequences.

        Returns:
            Union[float, tuple[float]]:
                Metric output.
        """
        raise NotImplementedError()
