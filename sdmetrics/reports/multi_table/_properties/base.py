"""Multi table base property class."""
import numpy as np
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

        all_details = []
        for table_name in metadata['tables']:
            property_instance = self._single_table_property()
            self._properties[table_name] = property_instance
            self._properties[table_name].get_score(
                real_data[table_name],
                synthetic_data[table_name],
                metadata['tables'][table_name],
                progress_bar
            )
            all_details.append(property_instance._details)

        self.is_computed = True
        all_details = pd.concat(all_details)
        return np.nanmean(all_details['Score'])

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
