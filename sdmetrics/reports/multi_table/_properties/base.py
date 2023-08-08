"""Multi table base property class."""
import numpy as np


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

    def _augment_error_msg(self, table_name):
        """Augment error messages in the details of a property.

        Column pair trends only generates an error message for the single table case.
        This method takes that error and augments it for the multi-table case.

        Args:
            table_name (str):
                Table name.
        """
        if 'Error' in self._properties[table_name]._details.columns:
            error_name = self._properties[table_name]._details['Error'][0][:25]
            if error_name == 'Error: ConstantInputError':
                new_error_msg = f"In table '{table_name}', t"
                errors = self._properties[table_name]._details['Error'].str
                self._properties[table_name]._details['Error'] = errors.replace('T', new_error_msg)

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

        average_score = np.zeros(len(metadata['tables']))
        for idx, table_name in enumerate(metadata['tables']):
            self._properties[table_name] = self._single_table_property()
            average_score[idx] = self._properties[table_name].get_score(
                real_data[table_name], synthetic_data[table_name], metadata['tables'][table_name],
                progress_bar
            )
            self._augment_error_msg(table_name)

        self.is_computed = True

        return np.nanmean(average_score)

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
