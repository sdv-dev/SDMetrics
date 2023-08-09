"""Multi table base property class."""
import pandas as pd


class BaseMultiTableProperty():
    """Base class for multi table properties.

    A property is a higher-level concept for a class that loops through all the base-level data
    and applies different base-level metrics based on the data type.

    Attributes:
        properties (dict):
            A dict mapping the table names to their single table properties.
        is_computed (bool):
            Whether or not the property has been computed.
        _only_multi_table (bool):
            Whether or not the property only exist for multi-tables.
        details_property (pandas.DataFrame):
            The multi table details property dataframe.
    """

    _single_table_property = None

    def __init__(self):
        self._properties = {}
        self.is_computed = False
        self._only_multi_table = False
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
        if not self._only_multi_table:
            details_frames = []
            for table_name in metadata['tables']:
                details = self._properties[table_name]._details.copy()
                details['Table'] = table_name
                details_frames.append(details)

            self.details_property = pd.concat(details_frames).reset_index(drop=True)
            cols = ['Table'] + [col for col in self.details_property if col != 'Table']
            self.details_property = self.details_property[cols]

    def _compute_average(self):
        """Average the scores for each column."""
        is_dataframe = isinstance(self.details_property, pd.DataFrame)
        has_score_column = 'Score' in self.details_property.columns
        if not is_dataframe or not has_score_column:
            raise ValueError("The property details must be a DataFrame with a 'Score' column.")

        return self.details_property['Score'].mean()

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

        return self._compute_average()

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
