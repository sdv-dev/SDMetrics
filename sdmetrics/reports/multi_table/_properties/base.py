"""Multi table base property class."""

import numpy as np
import pandas as pd


class BaseMultiTableProperty:
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
            try:
                return len(metadata['relationships'])
            except KeyError:
                return 0
        elif self._num_iteration_case == 'column_pair':
            num_columns = [len(table['columns']) for table in metadata['tables'].values()]
            return sum([(n_cols * (n_cols - 1)) // 2 for n_cols in num_columns])
        elif self._num_iteration_case == 'inter_table_column_pair':
            iterations = 0
            for relationship in metadata.get('relationships', []):
                parent_columns = metadata['tables'][relationship['parent_table_name']]['columns']
                child_columns = metadata['tables'][relationship['child_table_name']]['columns']
                iterations += len(parent_columns) * len(child_columns)
            return iterations

    @staticmethod
    def _extract_tuple(data, relation):
        parent_data = data[relation['parent_table_name']]
        child_data = data[relation['child_table_name']]
        return (
            parent_data[relation['parent_primary_key']],
            child_data[relation['child_foreign_key']],
        )

    def _compute_average(self):
        """Average the scores for each column."""
        is_dataframe = isinstance(self.details, pd.DataFrame)
        has_score_column = 'Score' in self.details.columns
        assert_message = "The property details must be in a DataFrame with a 'Score' column."

        assert is_dataframe, assert_message
        if not has_score_column:
            return np.nan

        return self.details['Score'].mean()

    def _generate_details(self, real_data, synthetic_data, metadata, progress_bar=None):
        """Generate the ``details`` dataframe for the multi-table property.

        Args:
            real_data (dict[str, pandas.DataFrame]):
                The real data.
            synthetic_data (dict[str, pandas.DataFrame]):
                The synthetic data.
            metadata (dict):
                The metadata, which contains each column's data type as well as relationships.
            progress_bar (tqdm.tqdm or None):
                The progress bar object. Defaults to None.
        """
        if self._single_table_property is None:
            raise NotImplementedError()

        for table_name, metadata_table in metadata['tables'].items():
            self._properties[table_name] = self._single_table_property()
            self._properties[table_name].get_score(
                real_data[table_name], synthetic_data[table_name], metadata_table, progress_bar
            )

        details_frames = []
        for table_name in metadata['tables']:
            details = self._properties[table_name].details.copy()
            details['Table'] = table_name
            details_frames.append(details)

        self.details = pd.concat(details_frames).reset_index(drop=True)

        cols = ['Table'] + [col for col in self.details if col != 'Table']
        self.details = self.details[cols]

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
        self._generate_details(real_data, synthetic_data, metadata, progress_bar)

        self.is_computed = True

        if 'Error' in self.details.columns and self.details['Error'].isna().all():
            self.details = self.details.drop('Error', axis=1)
        elif 'Error' in self.details.columns:
            self.details['Error'] = self.details['Error'].replace({np.nan: None})

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

    def get_details(self, table_name=None):
        """Return the details table for the property for the given table.

        Args:
            table_name (str):
                The name of the table to return details for.
                Defaults to None.

        Returns:
            pandas.DataFrame
        """
        if table_name is None:
            return self.details.copy()

        if self._num_iteration_case in ['relationship', 'inter_table_column_pair']:
            table_rows = (self.details['Parent Table'] == table_name) | (
                self.details['Child Table'] == table_name
            )
        else:
            table_rows = self.details['Table'] == table_name

        return self.details.loc[table_rows]
