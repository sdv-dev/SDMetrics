"""Multi table base property class."""


class BaseMultiTableProperty():
    """Base class for multi table properties.

    A property is a higher-level concept for a class that loops through all the base-level data
    and applies different base-level metrics based on the data type.

    Attributes:
        properties (dict):
            A dict mapping the table names to their single table properties.
    """

    _properties = None

    def get_score(self, real_data, synthetic_data, metadata, progress_bar):
        """Get the average score of all the individual metric scores computed.

        Args:
            real_data (pandas.DataFrame):
                The real data.
            synthetic_data (pandas.DataFrame):
                The synthetic data.
            metadata (dict):
                The metadata, which contains each column's data type as well as relationships.
            progress_bar (tqdm.tqdm):
                The progress bar object.

        Returns:
            float:
                The average score for the property for all the individual metric scores computed.
        """
        raise NotImplementedError()

    def get_visualization(self, table_name):
        """Return a visualization for each score in the property.

        Returns:
            plotly.graph_objects._figure.Figure
                The visualization for the property.
        """
        return self._properties[table_name].get_visualization()
