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
        is_computed (bool):
            Whether or not the property has been computed.
        details (pandas.DataFrame):
            The multi table details property dataframe.
    """

    _single_table_property = None
    _num_iteration_case = None

    def __init__(self):
        self._properties = {}
        self.is_computed = False
        self.details = pd.DataFrame()

    def _get_num_iterations(self, metadata):
        """Get the number of iterations for the property."""
        if self._num_iteration_case == 'column':
            return sum(len(metadata['tables'][table]['columns']) for table in metadata['tables'])
        elif self._num_iteration_case == 'table':
            return len(metadata['tables'])
        elif self._num_iteration_case == 'relationship':
            return len(metadata['relationships'])
        elif self._num_iteration_case == 'column_pair':
            num_columns = [len(table['columns']) for table in metadata['tables'].values()]
            return sum([(n_cols * (n_cols - 1)) // 2 for n_cols in num_columns])

    def _generate_details(self, metadata):
        """Generate the ``details`` dataframe for the multi-table property.

        This dataframe concatenates the ``_details`` dataframe of each single table property
        and adds a ``Table`` column to indicate which table the score is for.

        Args:
            metadata (dict):
                The metadata of the tables.
        """
        if not self._num_iteration_case == 'relationship':
            details_frames = []
            for table_name in metadata['tables']:
                details = self._properties[table_name].details.copy()
                details['Table'] = table_name
                details_frames.append(details)

            self.details = pd.concat(details_frames).reset_index(drop=True)

            if 'Error' in self.details.columns:
                self.details['Error'] = self.details['Error'].replace({np.nan: None})

            cols = ['Table'] + [col for col in self.details if col != 'Table']
            self.details = self.details[cols]

    def _compute_average(self):
        """Average the scores for each column."""
        is_dataframe = isinstance(self.details, pd.DataFrame)
        has_score_column = 'Score' in self.details.columns
        assert_message = "The property details must be a DataFrame with a 'Score' column."

        assert is_dataframe, assert_message
        assert has_score_column, assert_message

        return self.details['Score'].mean()

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

        self._generate_details(metadata)
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
