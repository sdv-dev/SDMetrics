"""Single table base property class."""
import pandas as pd


class BaseSingleTableProperty():
    """Base class for single table properties.

    A property is a higher-level concept for a class that loops through all the base-level data
    and applies different base-level metrics based on the data type.
    """

    _details = None

    def _compute_average(self):
        """Average the scores for each column."""
        if not isinstance(self._details, pd.DataFrame) or 'Score' not in self._details.columns:
            raise ValueError("The property details must be a DataFrame with a 'Score' column.")

        return round(self._details['Score'].mean(), 3)

    def _generate_details(self, real_data, synthetic_data, metadata, progress_bar=None):
        """Generate the _details dataframe for the property."""
        raise NotImplementedError()

    def get_score(self, real_data, synthetic_data, metadata, progress_bar=None):
        """Get the average score for the property on the data.

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
                The average score for the property.
        """
        self._details = self._generate_details(real_data, synthetic_data, metadata, progress_bar)
        return self._compute_average()

    def get_visualization(self):
        """Return a visualization for each score in the property.

        Returns:
            plotly.graph_objects._figure.Figure
                The visualization for the property.
        """
        raise NotImplementedError()
