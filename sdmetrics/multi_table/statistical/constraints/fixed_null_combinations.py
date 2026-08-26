"""Fixed Null Combinations Constraint."""

import pandas as pd

from sdmetrics.multi_table.statistical.constraints._utils import (
    _get_is_valid_dict,
    _is_list_of_type,
    _tuple_from_columns,
    compute_nans_column,
)
from sdmetrics.multi_table.statistical.constraints.base import BaseConstraint
from sdmetrics.multi_table.statistical.constraints.error import ConstraintNotApplicableError


class FixedNullCombinations(BaseConstraint):
    """Constraint to check the null combinations of across columns.

    Args:
        column_names (list[str]):
            Names of the columns that need to produce fixed combinations. Must
            contain at least two columns.
        table_name (str, optional):
            The name of the table that contains the columns. Optional if the
            data is only a single table. Defaults to None.
        fix_category_values (bool, optional):
            If True (default), also fix combinations of values in categorical/boolean
            columns included in `column_names`. If False, only fix nullness combinations.
    """

    def __init__(self, column_names, table_name=None, fix_category_values=True):
        super().__init__()

        if not isinstance(table_name, str):
            raise ValueError("The 'table_name' parameter must be a string.")

        if not _is_list_of_type(column_names, str):
            raise ValueError("The 'column_names' parameter must be a list of strings.")

        if len(column_names) < 2:
            raise ValueError('FixedNullCombinations constraint requires at least two columns.')

        if not isinstance(fix_category_values, bool):
            raise ValueError('`fix_category_values` must be a boolean.')

        self.table_name = table_name
        self.column_names = column_names
        self.fix_category_values = fix_category_values

        self._nan_combinations = frozenset()
        self._categorical_columns = []
        self._nan_fill_choices = {}

    def _validate_data(self, data, metadata=None):
        """Check that the table and all the referenced columns exist in the data."""
        if self.table_name not in data:
            raise ConstraintNotApplicableError(
                f"The table '{self.table_name}' is missing from the data."
            )

        columns = data[self.table_name].columns
        missing_columns = [
            column_name for column_name in self.column_names if column_name not in columns
        ]
        if missing_columns:
            missing_columns = "', '".join(missing_columns)
            raise ConstraintNotApplicableError(
                f"The column(s) '{missing_columns}' are missing from the table '{self.table_name}'."
            )

    def _assign_categorical_columns(self, metadata, table_name):
        self._categorical_columns = []
        for column in self.column_names:
            col_meta = metadata['tables'][table_name]['columns'].get(column, {})
            if col_meta.get('sdtype') in {'categorical', 'boolean'}:
                self._categorical_columns.append(column)

    @staticmethod
    def _get_nan_labels(nan_column, table_data):
        if nan_column is None:
            return pd.Series(['None'] * len(table_data), index=table_data.index, dtype='object')
        return nan_column

    def _assign_category_labels(self, unique_tuples):
        self._category_label_for_values = {}
        self._category_values_for_label = {}
        for idx, tup in enumerate(unique_tuples):
            label = f'fixed_combination#{idx}'
            self._category_label_for_values[tup] = label
            self._category_values_for_label[label] = {
                col_name: value for col_name, value in zip(self._categorical_columns, tup)
            }

    def _categorical_tuple_from_row(self, row):
        return _tuple_from_columns(row, self._categorical_columns)

    def _get_categorical_value_tuples(self, table_data):
        categorical_df = table_data[self._categorical_columns]
        return categorical_df.apply(self._categorical_tuple_from_row, axis=1)

    def _has_categorical_columns(self):
        """Check if there are categorical columns assigned."""
        return self.fix_category_values and len(self._categorical_columns) > 0

    @staticmethod
    def _normalize_nan_label(nan_label):
        if nan_label == 'None':
            return frozenset(['None'])
        return frozenset({col.strip() for col in nan_label.split(',')})

    def _group_category_combinations_by_nan(self, nan_labels_series, value_tuples):
        category_combinations_by_nanset = {}
        for row_index, cat_tuple in value_tuples.items():
            nan_label = nan_labels_series.loc[row_index]
            nan_set = self._normalize_nan_label(nan_label)
            if nan_set not in category_combinations_by_nanset:
                category_combinations_by_nanset[nan_set] = set()
            category_combinations_by_nanset[nan_set].add(cat_tuple)

        return category_combinations_by_nanset

    def _build_category_combinations(self, nan_column, table_data, metadata, table_name):
        """Build category combinations mapping if enabled."""
        nan_labels_series = self._get_nan_labels(nan_column, table_data)
        self._assign_categorical_columns(metadata, table_name)

        if self._has_categorical_columns():
            value_tuples = self._get_categorical_value_tuples(table_data)
            unique_tuples = pd.Series(value_tuples).unique().tolist()
            self._assign_category_labels(unique_tuples)
            self._category_combinations = frozenset(unique_tuples)
            self._category_combinations_by_nanset = self._group_category_combinations_by_nan(
                nan_labels_series, value_tuples
            )
        else:
            self._category_label_for_values = {}
            self._category_values_for_label = {}
            self._category_combinations = frozenset()
            self._category_combinations_by_nanset = {}

    def _fit(self, data, metadata=None):
        """Learn the null and value combinations that are present in the real data."""
        table_name = self._get_single_table_name(metadata)
        table_data = data[table_name]

        nan_column = compute_nans_column(table_data, self.column_names)
        if nan_column is not None:
            self._nan_combinations = frozenset([
                frozenset({col.strip() for col in combination.split(',')})
                for combination in nan_column.unique()
            ])
        else:
            self._nan_combinations = frozenset([frozenset(['None'])])

        self._build_category_combinations(nan_column, table_data, metadata, table_name)
        for column in self.column_names:
            if all(table_data[column].isna()):
                continue

            top_25_values = list(table_data[column].value_counts().head(25).keys())
            self._nan_fill_choices[column] = top_25_values

    def _validate_category_combinations(self, row, nan_cols):
        nan_valid = frozenset(nan_cols) in self._nan_combinations

        # If fixing categorical value combinations, also validate those combinations
        if self._has_categorical_columns():
            tup = _tuple_from_columns(row, self._categorical_columns)
            nan_set = frozenset(nan_cols)
            allowed = self._category_combinations_by_nanset.get(nan_set, set())
            cat_valid = tup in allowed

            return nan_valid and cat_valid

        return nan_valid

    def _is_valid(self, data, metadata=None):
        """Check that the data only has null combinations that are in the real data.

        Args:
            data (dict[str, pandas.DataFrame]):
                The data dictionary.
            metadata (dict):
                Metadata as a dictionary.

        Returns:
            dict[str, pandas.Series]:
                Whether each row is valid.
        """
        if not self._fitted:
            return _get_is_valid_dict(data, table_name=None)

        def is_valid_row(row):
            nan_cols = row[pd.isna(row)].keys()
            if len(nan_cols) == 0:
                nan_cols = ['None']

            return self._validate_category_combinations(row, nan_cols)

        table_name = self._get_single_table_name(metadata)
        is_valid = _get_is_valid_dict(data, table_name)
        data_columns = data[table_name][self.column_names]
        valid_rows = data_columns.apply(is_valid_row, axis=1)
        valid_rows.index = data[table_name].index
        is_valid[table_name] = valid_rows

        return is_valid
