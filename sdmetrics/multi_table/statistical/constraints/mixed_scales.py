"""Mixed Scales Constraint."""

import pandas as pd

from sdmetrics.multi_table.statistical.constraints._utils import (
    CustomNan,
    _get_is_valid_dict,
    _is_list_of_type,
)
from sdmetrics.multi_table.statistical.constraints.base import BaseConstraint
from sdmetrics.multi_table.statistical.constraints.error import ConstraintNotApplicableError
from sdmetrics.multi_table.statistical.constraints.utils.numerical_formatter import (
    NumericalFormatter,
)


class MixedScales(BaseConstraint):
    """Constraint to handle a continuous column with mixed scales.

    Args:
        mixed_scale_column_name (str):
            The name of the numerical column that can have different scales.
        segment_column_names (list[str]):
            A list of the names of one or more categorical (or boolean) columns. The combination
            values of these columns determine the segment for the mixed scale column.
        table_name (str, optional):
            Name of the table containing the mixed scale column and the segment columns.
            Optional if only using a single table. Default None.
    """

    def __init__(self, mixed_scale_column_name, segment_column_names, table_name=None):
        super().__init__()
        if table_name is not None and not isinstance(table_name, str):
            raise ValueError("The 'table_name' parameter must be a string.")

        if not isinstance(mixed_scale_column_name, str):
            raise ValueError('`mixed_scale_column_name` must be a string.')

        if not _is_list_of_type(segment_column_names):
            raise ValueError('`segment_column_names` must be a list of strings.')

        self.mixed_scale_column_name = mixed_scale_column_name
        self.segment_column_names = segment_column_names[:]
        self.table_name = table_name

        self._segment_column_name = None
        self._scaled_column_name = None
        self._segment_info = {}
        self._segment_tuple = ()
        self._missing_object = CustomNan()

    def _validate_data(self, data, metadata):
        table_name = self._get_single_table_name(self.metadata)
        columns_metadata = metadata['tables'][table_name]['columns']
        sdtype = columns_metadata.get(self.mixed_scale_column_name, {}).get('sdtype')
        if sdtype != 'numerical':
            raise ConstraintNotApplicableError(
                'A MixedScales constraint is being applied to columns with mismatched '
                f'sdtypes {self.mixed_scale_column_name}. The mixed_scale_column must be numerical.'
            )

        for column_name in self.segment_column_names:
            sdtype = columns_metadata.get(column_name, {}).get('sdtype')
            if sdtype not in ['categorical', 'boolean']:
                raise ConstraintNotApplicableError(
                    'A MixedScales constraint is being applied to segment columns with '
                    f'mismatched sdtypes {column_name}. All segment columns must be '
                    'categorical.'
                )

    def _replace_nans(self, segment):
        if isinstance(segment, tuple):
            return tuple(self._missing_object if pd.isna(val) else val for val in segment)

        return self._missing_object if pd.isna(segment) else segment

    def _fit(self, data, metadata):
        """Fit the constraint."""
        table_name = self._get_single_table_name(metadata)
        table_data = data[table_name]
        grouped = table_data.groupby(self.segment_column_names, dropna=False)
        segment_list = []
        self._segment_tuple = ()
        self._segment_info = {}

        for idx, (segment, group) in enumerate(grouped):
            # add segment to list before tuple to string conversion so indexing works
            segment = self._replace_nans(segment)
            segment_list.append(segment)

            if isinstance(segment, tuple) and len(segment) == 1:
                segment = segment[0]

            segment_formatter = NumericalFormatter(
                enforce_rounding=True, enforce_min_max_values=True
            )
            segment_formatter.learn_format(group[self.mixed_scale_column_name])
            segment_null_pctg = group[self.mixed_scale_column_name].isna().sum() / group.shape[0]
            self._segment_info[segment] = {
                'formatter': segment_formatter,
                'max': segment_formatter._max_value,
                'min': segment_formatter._min_value,
                'null_proportion': segment_null_pctg,
                'index': idx,
            }

        self._segment_tuple = tuple(segment_list)

    def _is_valid(self, data, metadata=None):
        """Check that the data respect the bounds of the segment.

        Args:
            data (dict[str, pandas.DataFrame]):
                The data dictionary.
            metadata (dict):
                Metadata as a dictionary.

        Returns:
            dict[str, pandas.Series]:
                For each table, a pandas Series indicating whether each row is valid.
        """
        if not self._fitted:
            return _get_is_valid_dict(data, table_name=None)

        table_name = self._get_single_table_name(self.metadata)
        is_valid_data = _get_is_valid_dict(data, table_name)

        def _ge_min(numeric_col):
            segment = self._replace_nans(numeric_col.name)
            if segment not in self._segment_info:
                return pd.Series(False, index=numeric_col.index)

            return pd.isna(numeric_col) | (numeric_col >= self._segment_info[segment]['min'])

        def _le_max(numeric_col):
            segment = self._replace_nans(numeric_col.name)
            if segment not in self._segment_info:
                return pd.Series(False, index=numeric_col.index)

            return pd.isna(numeric_col) | (numeric_col <= self._segment_info[segment]['max'])

        table_data = data[table_name][self.segment_column_names + [self.mixed_scale_column_name]]
        table_data = table_data.groupby(self.segment_column_names, dropna=False)
        valid_rows = table_data[self.mixed_scale_column_name].transform(_ge_min) & table_data[
            self.mixed_scale_column_name
        ].transform(_le_max)

        is_valid_data[table_name] = valid_rows
        return is_valid_data
