"""Unified table base report."""

import pandas as pd

from sdmetrics._utils_metadata import _validate_metadata
from sdmetrics.reports.multi_table.base_multi_table_report import BaseMultiTableReport


class BaseUnifiedReport(BaseMultiTableReport):
    """Base Unified Report for single-table and multi-table data."""

    @staticmethod
    def _normalize_data(data, metadata):
        """Normalize a pandas DataFrame to a dictionary of tables."""
        if isinstance(data, pd.DataFrame):
            table_names = list(metadata.get('tables', []))
            if len(table_names) == 0:
                raise ValueError(
                    'The metadata is empty. Please add exactly one table to the metadata '
                    'when passing a pandas.DataFrame.'
                )
            if len(table_names) > 1:
                raise ValueError(
                    'A single pandas.DataFrame can only be used with metadata that contains '
                    f'one table. Your metadata contains multiple tables: {table_names}. '
                    'Please either remove the extra tables from the metadata or pass the data '
                    'as a dictionary with one pandas.DataFrame per table.'
                )
            return {table_names[0]: data}
        return data

    def _validate_data_format(self, real_data, synthetic_data):
        """Validate that the real and synthetic data have compatible formats.

        Args:
            real_data (pandas.DataFrame or dict[str, pd.DataFrame]):
                The real data.
            synthetic_data (pandas.DataFrame or dict[str, pd.DataFrame]):
                The synthetic data.
        """
        is_real_dict = isinstance(real_data, dict)
        is_real_dataframe = isinstance(real_data, pd.DataFrame)

        is_synthetic_dict = isinstance(synthetic_data, dict)
        is_synthetic_dataframe = isinstance(synthetic_data, pd.DataFrame)

        if is_real_dict and is_synthetic_dict:
            all_real_dataframes = all(
                isinstance(table, pd.DataFrame) for table in real_data.values()
            )
            all_synthetic_dataframes = all(
                isinstance(table, pd.DataFrame) for table in synthetic_data.values()
            )
            if all_real_dataframes and all_synthetic_dataframes:
                return
        elif is_real_dataframe and is_synthetic_dataframe:
            return

        error_message = (
            f'{self.__class__.__name__} expects real_data and synthetic_data to both be '
            'pandas.DataFrame, or both be dictionaries mapping table names to '
            f'pandas.DataFrame. Received real_data={type(real_data).__name__} and '
            f'synthetic_data={type(synthetic_data).__name__}.'
        )

        raise ValueError(error_message)

    def _validate(self, real_data, synthetic_data, metadata):
        """Validate the inputs and normalize the real_data/synthetic_data.

        Args:
            real_data (pandas.DataFrame or dict[str, pd.DataFrame]):
                The real data.
            synthetic_data (pandas.DataFrame or dict[str, pd.DataFrame]):
                The synthetic data.
            metadata (dict):
                The metadata.
        """
        _validate_metadata(metadata)
        self.table_names = list(metadata.get('tables', []))
        self._validate_data_format(real_data, synthetic_data)

        real_data = self._normalize_data(real_data, metadata)
        synthetic_data = self._normalize_data(synthetic_data, metadata)
        self._validate_metadata_matches_data(
            real_data,
            synthetic_data,
            metadata,
        )
