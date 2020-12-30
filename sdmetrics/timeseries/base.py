"""Base Time Series metric class."""

from operator import attrgetter

from sdmetrics.base import BaseMetric


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
            'type': 'numerical',
            'subtype': 'integer',
        },
        'f': {
            'type': 'numerical',
            'subtype': 'float',
        },
        'O': {
            'type': 'categorical',
        },
        'b': {
            'type': 'boolean',
        },
        'M': {
            'type': 'datetime',
        }
    }

    @classmethod
    def _validate_inputs(cls, real_data, synthetic_data, metadata=None, entity_columns=None):
        if set(real_data.columns) != set(synthetic_data.columns):
            raise ValueError('`real_data` and `synthetic_data` must have the same columns')

        if metadata is not None:
            if not isinstance(metadata, dict):
                metadata = metadata.to_dict()

            fields = metadata['fields']
            for column in real_data.columns:
                if column not in fields:
                    raise ValueError(f'Column {column} not found in metadata')

            for field in fields.keys():
                if field not in real_data.columns:
                    raise ValueError(f'Field {field} not found in data')

        else:
            dtype_kinds = real_data.dtypes.apply(attrgetter('kind'))
            metadata = {'fields': dtype_kinds.apply(cls._DTYPES_TO_TYPES.get).to_dict()}

        entity_columns = metadata.get('entity_columns', entity_columns or [])

        return metadata, entity_columns

    @classmethod
    def compute(cls, real_data, synthetic_data, metadata=None, entity_columns=None):
        """Compute this metric.

        Args:
            real_data (pandas.DataFrame):
                The values from the real dataset, passed as a pandas.DataFrame.
            synthetic_data (pandas.DataFrame):
                The values from the synthetic dataset, passed as a pandas.DataFrame.
            metadata (dict):
                TimeSeries metadata dict. If not passed, it is build based on the
                real_data fields and dtypes.
            entity_columns (list[str]):
                Names of the columns which identify different time series
                sequences.

        Returns:
            Union[float, tuple[float]]:
                Metric output.
        """
        raise NotImplementedError()
