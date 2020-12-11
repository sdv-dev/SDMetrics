"""Base Single Table metric class."""

from operator import attrgetter

from sdmetrics.base import BaseMetric


class SingleTableMetric(BaseMetric):
    """Base class for metrics that apply to single tables.

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
    def _select_fields(cls, metadata, types):
        fields = []
        if isinstance(types, str):
            types = (types, )

        for field_name, field_meta in metadata['fields'].items():
            field_type = field_meta['type']
            field_subtype = field_meta.get('subtype')
            if any(t in types for t in (field_type, (field_type, ), (field_type, field_subtype))):
                fields.append(field_name)

        return fields

    @classmethod
    def _validate_inputs(cls, real_data, synthetic_data, metadata=None):
        if set(real_data.columns) != set(synthetic_data.columns):
            raise ValueError('`real_data` and `synthetic_data` must have the same columns')

        if metadata is not None:
            fields = metadata['fields']
            for column in real_data.columns:
                if column not in fields:
                    raise ValueError(f'Column {column} not found in metadata')

            for field in fields.keys():
                if field not in real_data.columns:
                    raise ValueError(f'Field {field} not found in data')

            return metadata

        dtype_kinds = real_data.dtypes.apply(attrgetter('kind'))
        return {'fields': dtype_kinds.apply(cls._DTYPES_TO_TYPES.get).to_dict()}

    @classmethod
    def compute(cls, real_data, synthetic_data, metadata=None):
        """Compute this metric.

        Args:
            real_data (pandas.DataFrame):
                The values from the real dataset, passed as a pandas.DataFrame.
            synthetic_data (pandas.DataFrame):
                The values from the synthetic dataset, passed as a pandas.DataFrame.
            metadata (dict):
                Table metadata dict. If not passed, it is build based on the
                real_data fields and dtypes.

        Returns:
            Union[float, tuple[float]]:
                Metric output.
        """
        raise NotImplementedError()
