"""Unified table base report."""

import pandas as pd

from sdmetrics._utils_metadata import _validate_metadata
from sdmetrics.reports.multi_table.base_multi_table_report import BaseMultiTableReport


class BaseUnifiedReport(BaseMultiTableReport):
    """Base Unified Report for single-table and multi-table data."""

    def _validate_data_format(self, real_data, synthetic_data):
        """Validate that the real and synthetic data have compatible formats.

        Args:
            real_data (dict[str, pd.DataFrame]):
                The real data.
            synthetic_data (dict[str, pd.DataFrame]):
                The synthetic data.
        """
        if not isinstance(real_data, dict):
            raise ValueError(
                'Please pass in a dictionary mapping tables to dataframes for real_data.'
            )

        if not isinstance(synthetic_data, dict):
            raise ValueError(
                'Please pass in a dictionary mapping tables to dataframes for synthetic_data.'
            )

        all_real_dataframes = all(isinstance(table, pd.DataFrame) for table in real_data.values())
        all_synthetic_dataframes = all(
            isinstance(table, pd.DataFrame) for table in synthetic_data.values()
        )
        if all_real_dataframes and all_synthetic_dataframes:
            return

        raise ValueError(
            f'{self.__class__.__name__} expects real_data and synthetic_data to both be '
            'pandas.DataFrame, or both be dictionaries mapping table names to '
            f'pandas.DataFrame. Received real_data={type(real_data).__name__} and '
            f'synthetic_data={type(synthetic_data).__name__}.'
        )

    def _validate(self, real_data, synthetic_data, metadata):
        """Validate the inputs.

        Args:
            real_data (dict[str, pd.DataFrame]):
                The real data.
            synthetic_data (dict[str, pd.DataFrame]):
                The synthetic data.
            metadata (dict):
                The metadata.
        """
        _validate_metadata(metadata)
        self.table_names = list(metadata.get('tables', []))
        self._validate_data_format(real_data, synthetic_data)
        self._validate_metadata_matches_data(
            real_data,
            synthetic_data,
            metadata,
        )
