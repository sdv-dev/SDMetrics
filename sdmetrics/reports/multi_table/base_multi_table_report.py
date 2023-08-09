"""Single table base property class."""
from sdmetrics.reports.single_table.base_report import BaseReport


class BaseMultiTableReport(BaseReport):
    """Base report class for multi table reports.

    Attributes:
        table_names (list):
            A list of the table names.
    """

    def __init__(self):
        super().__init__()
        self.table_names = []

    def _validate_relationships(self, real_data, synthetic_data, metadata):
        """Validate that the relationships are valid."""
        for rel in metadata.get('relationships', []):
            parent_dtype = real_data[rel['parent_table_name']][rel['parent_primary_key']].dtype
            child_dtype = real_data[rel['child_table_name']][rel['child_foreign_key']].dtype
            if (parent_dtype == 'object' and child_dtype != 'object') or (
                    parent_dtype != 'object' and child_dtype == 'object'):
                parent = rel['parent_table_name']
                parent_key = rel['parent_primary_key']
                child = rel['child_table_name']
                child_key = rel['child_foreign_key']
                error_msg = (
                    f"The '{parent}' table and '{child}' table cannot be merged. Please "
                    f"make sure the primary key in '{parent}' ('{parent_key}') and the "
                    f"foreign key in '{child}' ('{child_key}') have the same data type."
                )
                raise ValueError(error_msg)

    def _validate_metadata_matches_data(self, real_data, synthetic_data, metadata):
        """Validate that the metadata matches the data."""
        self.table_names = list(metadata['tables'].keys())
        for table in self.table_names:
            super()._validate_metadata_matches_data(
                real_data[table], synthetic_data[table], metadata['tables'][table]
            )

        self._validate_relationships(real_data, synthetic_data, metadata)

    def _check_table_names(self, table_name):
        if table_name not in self.table_names:
            raise ValueError(f"Unknown table ('{table_name}'). Must be one of {self.table_names}.")

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
            return self._properties[property_name].details_property.copy()

        self._check_table_names(table_name)

        if not self._properties[property_name]._only_multi_table:
            table_rows = self._properties[property_name].details_property['Table'] == table_name
            details = self._properties[property_name].details_property.loc[table_rows]
            details = details.drop(columns=['Table'])
        else:
            details = self._properties[property_name].get_details(table_name)

        return details.copy()

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
        if table_name is None:
            raise ValueError(
                'Please provide a table name to get a visualization for the property.'
            )

        self._validate_property_generated(property_name)
        self._check_table_names(table_name)
        return self._properties[property_name].get_visualization(table_name)
