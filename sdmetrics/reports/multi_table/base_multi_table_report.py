"""Single table base property class."""

import pandas as pd

from sdmetrics.reports.base_report import BaseReport
from sdmetrics.visualization import set_plotly_config


class BaseMultiTableReport(BaseReport):
    """Base report class for multi table reports.

    Attributes:
        table_names (list):
            A list of the table names.
    """

    def __init__(self):
        super().__init__()
        self.table_names = []

    def _validate_data_format(self, real_data, synthetic_data):
        """Validate that the real and synthetic are dictionnaries of tables."""
        is_real_dict = isinstance(real_data, dict)
        is_synthetic_dict = isinstance(synthetic_data, dict)
        if is_real_dict and is_synthetic_dict:
            all_real_dataframes = all(
                isinstance(table, pd.DataFrame) for table in real_data.values()
            )
            all_synthetic_dataframes = all(
                isinstance(table, pd.DataFrame) for table in synthetic_data.values()
            )
            if all_real_dataframes and all_synthetic_dataframes:
                return

        error_message = (
            f'Multi table {self.__class__.__name__} expects real and synthetic data to be'
            ' dictionaries of pandas.DataFrame. If your real and synthetic data are pd.DataFrame,'
            f' please use the single-table {self.__class__.__name__} instead.'
        )

        raise ValueError(error_message)

    def _validate_relationships(self, real_data, synthetic_data, metadata):
        """Validate that the relationships are valid."""
        for rel in metadata.get('relationships', []):
            parent_dtype = real_data[rel['parent_table_name']][rel['parent_primary_key']].dtype
            child_dtype = real_data[rel['child_table_name']][rel['child_foreign_key']].dtype
            if (parent_dtype == 'object' and child_dtype != 'object') or (
                parent_dtype != 'object' and child_dtype == 'object'
            ):
                parent = rel['parent_table_name']
                parent_key = rel['parent_primary_key']
                child = rel['child_table_name']
                child_key = rel['child_foreign_key']
                error_msg = (
                    f"The '{parent}' table and '{child}' table cannot be merged "
                    'for computing the cardinality. Please make sure the primary key'
                    f" in '{parent}' ('{parent_key}') and the foreign key in '{child}'"
                    f" ('{child_key}') have the same data type."
                )
                raise ValueError(error_msg)

    def _validate_metadata_matches_data(self, real_data, synthetic_data, metadata):
        """Validate that the metadata matches the data."""
        for table in self.table_names:
            super()._validate_metadata_matches_data(
                real_data[table], synthetic_data[table], metadata['tables'][table]
            )

        self._validate_relationships(real_data, synthetic_data, metadata)

    def generate(self, real_data, synthetic_data, metadata, verbose=True):
        """Generate report.

        This method generates the report by iterating through each property and calculating
        the score for each property.

        Args:
            real_data (pandas.DataFrame):
                The real data.
            synthetic_data (pandas.DataFrame):
                The synthetic data.
            metadata (dict):
                The metadata, which contains each column's data type as well as relationships.
            verbose (bool):
                Whether or not to print report summary and progress.
        """
        results = super().generate(real_data, synthetic_data, metadata, verbose)
        self.table_names = list(metadata.get('tables', {}).keys())

        return results

    def _check_table_names(self, table_name):
        if table_name not in self.table_names:
            raise ValueError(f"Unknown table ('{table_name}'). Must be one of {self.table_names}.")

    @staticmethod
    def convert_datetimes(real_data, synthetic_data, metadata):
        """Try to convert all datetime columns to datetime dtype.

        Args:
            real_data (pandas.DataFrame):
                The real data.
            synthetic_data (pandas.DataFrame):
                The synthetic data.
            metadata (dict):
                The metadata, which contains each column's data type as well as relationships.
        """
        for table, table_metadata in metadata['tables'].items():
            BaseReport.convert_datetimes(real_data[table], synthetic_data[table], table_metadata)

    def get_details(self, property_name, table_name=None):
        """Return the details table for the given property name.

        Args:
            property_name (str):
                The name of the property to return details for.
            table_name (str):
                The name of the table to return details for.
                Defaults to None.

        Returns:
            pandas.DataFrame
        """
        self._validate_property_generated(property_name)
        if not table_name:
            return self._properties[property_name].details.copy()

        self._check_table_names(table_name)

        details = self._properties[property_name].get_details(table_name)
        return details.copy()

    @set_plotly_config
    def get_visualization(self, property_name, table_name=None):
        """Return a visualization for the given property and table_name.

        Args:
            property_name (str):
                The name of the property.
            table_name (str):
                The name of the table.
                Defaults to None.

        Returns:
            plotly.graph_objects._figure.Figure
                The visualization for the requested property.
        """
        if property_name == 'Data Structure':
            return self._properties[property_name].get_visualization(table_name)

        if table_name is None:
            raise ValueError('Please provide a table name to get a visualization for the property.')

        self._validate_property_generated(property_name)
        self._check_table_names(table_name)
        return self._properties[property_name].get_visualization(table_name)
