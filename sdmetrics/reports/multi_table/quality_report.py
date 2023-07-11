"""Multi table quality report."""

import pickle
import warnings

import pandas as pd
import pkg_resources

from sdmetrics.reports.multi_table._properties import Cardinality, ColumnPairTrends, ColumnShapes
from sdmetrics.reports.utils import validate_multi_table_inputs


class QualityReport():
    """Multi table quality report.

    This class creates a quality report for multi-table data. It calculates the quality
    score along three properties - Column Shapes, Column Pair Trends, and Cardinality.
    """

    def __init__(self):
        self._tables = []
        self._overall_quality_score = None
        self._properties_instances = {}
        self._properties_scores = {}
        self._is_generated = False
        self._package_version = None

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
                NOTE: todo
        """
        validate_multi_table_inputs(real_data, synthetic_data, metadata)

        self._tables = list(real_data.keys())

        self._properties_instances = {
            'Column Shapes': ColumnShapes(),
            'Column Pair Trends': ColumnPairTrends(),
            'Cardinality': Cardinality()
        }

        self._properties_scores = {
            property_name: property_instance.get_score(real_data, synthetic_data, metadata)
            for property_name, property_instance in self._properties_instances.items()
        }

        scores = self._properties_scores.values()
        self._overall_quality_score = sum(scores) / len(scores)
        self._is_generated = True

    def _validate_generated(self):
        if not self._is_generated:
            raise ValueError(
                "The report has not been generated yet. Please call the 'generate' method.")

    def get_score(self):
        """Return the overall quality score.

        Returns:
            float
                The overall quality score.
        """
        self._validate_generated()

        return self._overall_quality_score

    def get_properties(self):
        """Return the score for each property.

        Returns:
            pandas.DataFrame
                The property score breakdown.
        """
        self._validate_generated()

        return pd.DataFrame({
            'Property': self._properties_scores.keys(),
            'Score': self._properties_scores.values(),
        })

    def _validate_inputs(self, property_name, table_name):
        self._validate_generated()

        valid_properties = list(self._properties_instances.keys())
        if property_name not in valid_properties:
            raise ValueError(
                f"Invalid property name ('{property_name}'). "
                f'It must be one of {valid_properties}.'
            )

        if (table_name is not None) and (table_name not in self._tables):
            raise ValueError(f"Unknown table ('{table_name}'). Must be one of {self._tables}.")

        if property_name in ['Column Shapes', 'Column Pair Trends'] and table_name is None:
            raise ValueError('Table name must be provided when viewing details for '
                             f"property '{property_name}'.")

    def get_visualization(self, property_name, table_name=None):
        """Return a visualization for each score for the given property and table.

        Args:
            property_name (str):
                The name of the property to return score details for.
            table_name (str):
                The table to show scores for. Must be provided for 'Column Shapes'
                and 'Column Pair Trends'.

        Returns:
            plotly.graph_objects._figure.Figure
                A visualization of the requested property's scores.
        """
        self._validate_inputs(property_name, table_name)

        return self._properties_instances[property_name].get_visualization(table_name)

    def get_details(self, property_name, table_name=None):
        """Return the details for each score for the given property name.

        Args:
            property_name (str):
                The name of the property to return score details for.
            table_name (str):
                Optionally filter results by table. Must be provided for 'Column Shapes'
                and 'Column Pair Trends'.

        Returns:
            pandas.DataFrame
                The score breakdown.
        """
        self._validate_inputs(property_name, table_name)

        if property_name != 'Cardinality':
            return self._properties_instances[property_name]._properties[table_name]._details.copy(
            )

        details = self._properties_instances[property_name]._details.copy()
        if table_name:
            return {k: v for k, v in details.items() if k[0] == table_name or k[1] == table_name}

        return details

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
