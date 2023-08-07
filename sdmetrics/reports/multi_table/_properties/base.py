"""Multi table base property class."""
import pandas as pd


class BaseMultiTableProperty():
    """Base class for multi table properties.

    A property is a higher-level concept for a class that loops through all the base-level data
    and applies different base-level metrics based on the data type.

    Attributes:
        properties (dict):
            A dict mapping the table names to their single table properties.
    """

    _single_table_property = None

    def __init__(self):
        self._properties = {}
        self.is_computed = False
        self.details_property = pd.DataFrame()

    def _get_num_iterations(self, metadata):
        """Get the number of iterations for the property."""
        raise NotImplementedError()

    def _generate_details_property(self, metadata):
        """Generate the ``details_property`` dataframe for the multi-table property.

        This dataframe concatenates the ``_details`` dataframe of each single table property
        and adds a ``Table`` column to indicate which table the score is for.

        Args:
            metadata (dict):
                The metadata of the tables.
        """
        for table_name in metadata['tables']:
            details = self._properties[table_name]._details.copy()
            details['Table'] = table_name
            self.details_property = pd.concat(
                [self.details_property, details]
            )

    def get_score(self, real_data, synthetic_data, metadata, progress_bar=None):
        """Get the average score of all the individual metric scores computed.

        Args:
            real_data (pandas.DataFrame):
                The real data.
            synthetic_data (pandas.DataFrame):
                The synthetic data.
            metadata (dict):
                The metadata, which contains each column's data type as well as relationships.
            progress_bar (tqdm.tqdm or None):
                The progress bar object. Defaults to None.

        Returns:
            float:
                The average score for the property for all the individual metric scores computed.
        """
        if self._single_table_property is None:
            raise NotImplementedError()

        for table_name, metadata_table in metadata['tables'].items():
            self._properties[table_name] = self._single_table_property()
            self._properties[table_name].get_score(
                real_data[table_name], synthetic_data[table_name], metadata_table,
                progress_bar
            )

        self._generate_details_property(metadata)
        self.is_computed = True

        return self.details_property['Score'].mean()

    def get_visualization(self, table_name):
        """Return a visualization for each score in the property.

        Args:
            table_name (str):
                Table name to get the visualization for.

        Returns:
            plotly.graph_objects._figure.Figure
                The visualization for the property.
        """
        if not self.is_computed:
            raise ValueError(
                'The property must be computed before getting a visualization.'
                'Please call the ``get_score`` method first.'
            )

        return self._properties[table_name].get_visualization()
