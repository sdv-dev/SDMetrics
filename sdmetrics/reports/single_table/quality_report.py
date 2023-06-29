"""Single table quality report."""
import pickle
import sys
import warnings

import numpy as np
import pandas as pd
import pkg_resources

from sdmetrics.reports.single_table._properties import ColumnPairTrends, ColumnShapes
from sdmetrics.reports.utils import _validate_categorical_values


class QualityReport():
    """Single table quality report.

    This class creates a quality report for single-table data. It calculates the quality
    score along two properties - Column Shapes and Column Pair Trends.
    """

    def __init__(self):
        self._overall_quality_score = None
        self.is_generated = False
        self._properties = {
            'Column Shapes': ColumnShapes(),
            'Column Pair Trends': ColumnPairTrends()
        }

    def _validate_metadata_matches_data(self, real_data, synthetic_data, metadata):
        """Validate that the metadata matches the data.

        Raise an error if the column metadata does not match the column data.

        Args:
            real_data (pandas.DataFrame):
                The real data.
            synthetic_data (pandas.DataFrame):
                The synthetic data.
            metadata (dict):
                The metadata of the table.
        """
        real_columns = set(real_data.columns)
        synthetic_columns = set(synthetic_data.columns)
        metadata_columns = set(metadata['columns'].keys())

        missing_data = metadata_columns.difference(real_columns.union(synthetic_columns))
        missing_metadata = real_columns.union(synthetic_columns).difference(metadata_columns)
        missing_columns = missing_data.union(missing_metadata)

        if missing_columns:
            error_message = (
                'The metadata does not match the data. The following columns are missing'
                ' in the real/synthetic data or in the metadata: '
                f"{', '.join(sorted(missing_columns))}"
            )
            raise ValueError(error_message)

    def validate(self, real_data, synthetic_data, metadata):
        """Validate the inputs.

        Args:
            real_data (pandas.DataFrame):
                The real data.
            synthetic_data (pandas.DataFrame):
                The synthetic data.
            metadata (dict):
                The metadata of the table.
        """
        if not isinstance(metadata, dict):
            metadata = metadata.to_dict()

        self._validate_metadata_matches_data(real_data, synthetic_data, metadata)
        _validate_categorical_values(real_data, synthetic_data, metadata)

    def _print_results(self, out=sys.stdout):
        """Print the quality report results."""
        out.write(
                f'\nOverall Quality Score: {round(self._overall_quality_score * 100, 2)}%\n\n'
        )
        out.write('Properties:\n')

        for property_name in self._properties:
            property_score = self._properties[property_name]._compute_average()
            out.write(
                f'- {property_name}: {property_score * 100}%\n'
            )

    def generate(self, real_data, synthetic_data, metadata, verbose=True):
        """Generate report.

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
        self.validate(real_data, synthetic_data, metadata)

        scores = []
        for property_name in self._properties:
            scores.append(self._properties[property_name].get_score(
                real_data, synthetic_data, metadata)
            )

        self._overall_quality_score = np.nanmean(scores)
        self.is_generated = True

        if verbose:
            self._print_results()

    def _validate_property_generated(self, property_name):
        """Validate that the given property name and that the report has been generated."""
        if property_name not in ['Column Shapes', 'Column Pair Trends']:
            raise ValueError(
                f"Invalid property name '{property_name}'."
                " Valid property names are 'Column Shapes' and 'Column Pair Trends'."
            )

        if not self.is_generated:
            raise ValueError(
                'Quality report must be generated before getting details. Call `generate` first.'
            )

    def get_score(self):
        """Return the overall quality score.

        Returns:
            float
                The overall quality score.
        """
        return self._overall_quality_score

    def get_properties(self):
        """Return the property score.

        Returns:
            pandas.DataFrame
                The property score.
        """
        name, score = [], []
        for property_name in self._properties:
            name.append(property_name)
            score.append(self._properties[property_name]._compute_average())
        return pd.DataFrame({
            'Property': name,
            'Score': score,
        })

    def get_visualization(self, property_name):
        """Return a visualization for each score for the given property name.

        Args:
            property_name (str):
                The name of the property to return score details for.

        Returns:
            plotly.graph_objects._figure.Figure
                The visualization for the requested property.
        """
        self._validate_property_generated(property_name)

        return self._properties[property_name].get_visualization()

    def get_details(self, property_name):
        """Return the details table for the given property name.

        Args:
            property_name (str):
                The name of the property to return details for.

        Returns:
            pandas.DataFrame
        """
        self._validate_property_generated(property_name)

        return self._properties[property_name]._details.copy()

    def save(self, filepath):
        """Save this report instance to the given path using pickle.

        Args:
            filepath (str):
                The path to the file where the report instance will be serialized.
        """
        self._package_version = pkg_resources.get_distribution('sdmetrics').version

        with open(filepath, 'wb') as output:
            pickle.dump(self, output)

    @classmethod
    def load(cls, filepath):
        """Load a ``QualityReport`` instance from a given path.

        Args:
            filepath (str):
                The path to the file where the report is stored.

        Returns:
            QualityReort:
                The loaded quality report instance.
        """
        current_version = pkg_resources.get_distribution('sdmetrics').version

        with open(filepath, 'rb') as f:
            report = pickle.load(f)
            if current_version != report._package_version:
                warnings.warn(
                    f'The report was created using SDMetrics version `{report._package_version}` '
                    f'but you are currently using version `{current_version}`. '
                    'Some features may not work as intended.')

            return report
