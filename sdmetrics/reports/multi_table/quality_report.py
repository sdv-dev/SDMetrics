"""Multi table quality report."""

import pickle
import sys
import warnings

import numpy as np
import pandas as pd
import pkg_resources
import tqdm

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
        self._property_errors = {}

    def _print_results(self, out=sys.stdout):
        """Print the quality report results."""
        if pd.isna(self._overall_quality_score) & any(self._property_errors.values()):
            out.write('\nOverall Quality Score: Error computing report.\n\n')
        else:
            out.write(
                f'\nOverall Quality Score: {round(self._overall_quality_score * 100, 2)}%\n\n')

        out.write('Properties:\n')

        for property_name, score in self._properties_scores.items():
            if not pd.isna(score):
                out.write(f'{property_name}: {round(score * 100, 2)}%\n')
            elif property_name in self._property_errors:
                out.write(f'{property_name}: Error computing property.\n')
            else:
                out.write(f'{property_name}: NaN\n')

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
                Whether or not to print the report summary and progress.
        """
        validate_multi_table_inputs(real_data, synthetic_data, metadata)

        self._tables = list(real_data.keys())

        self._properties_instances = {
            'Column Shapes': ColumnShapes(),
            'Column Pair Trends': ColumnPairTrends(),
            'Cardinality': Cardinality()
        }

        if verbose:
            sys.stdout.write('Generating report ...\n')

        num_columns = [len(table['columns']) for table in metadata['tables'].values()]
        num_properties = len(self._properties_instances)
        progress_bar = None
        for index, property_tuple in enumerate(self._properties_instances.items()):
            property_name, property_instance = property_tuple
            if verbose:
                if property_name == 'Column Shapes':
                    num_iterations = sum(num_columns)
                elif property_name == 'Column Pair Trends':
                    # for each table, the number of combinations of pairs of columns is
                    # n * (n - 1) / 2, where n is the number of columns in the table
                    num_iterations = sum([(n_cols * (n_cols - 1)) // 2 for n_cols in num_columns])
                elif property_name == 'Cardinality':
                    num_iterations = len(metadata['relationships'])

                progress_bar = tqdm.tqdm(total=num_iterations, file=sys.stdout)
                progress_bar.set_description(
                    f'({index + 1}/{num_properties}) Evaluating {property_name}: ')

            try:
                self._properties_scores[property_name] = property_instance.get_score(
                    real_data, synthetic_data, metadata, progress_bar)
            except BaseException:
                self._properties_scores[property_name] = np.nan
                self._property_errors[property_name] = True

            if verbose:
                progress_bar.close()

        scores = list(self._properties_scores.values())
        self._overall_quality_score = np.nanmean(scores)
        self._is_generated = True

        if verbose:
            self._print_results(sys.stdout)

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
                The score for each property.
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

    def _validate_visualization(self, property_name, table_name):
        self._validate_inputs(property_name, table_name)
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
        self._validate_visualization(property_name, table_name)

        return self._properties_instances[property_name].get_visualization(table_name)

    def _get_details_non_cardinality(self, property_instance, table_name):
        if table_name:
            details = {table_name: property_instance._properties[table_name]._details.copy()}
        else:
            details = {
                table_name: property_._details
                for table_name, property_ in property_instance._properties.items()
            }

        # Add a column with the table name for each details
        for table_name in details:
            table_column = pd.DataFrame({'Table': [table_name] * len(details[table_name])})
            details[table_name] = pd.concat([table_column, details[table_name]], axis=1)

        return pd.concat(list(details.values()), ignore_index=True)

    def _get_details_cardinality(self, property_instance, table_name):
        # For Cardinality, the details are a dictionary where the keys are tuples (table1, table2).
        # If table_name is passed, select only the tuples which contain it.
        details = property_instance._details
        if table_name:
            details = {
                table_names: detail
                for table_names, detail in details.items()
                if table_name in table_names
            }

        details_dataframe = pd.DataFrame()
        for tables, scores in details.items():
            new_row = pd.DataFrame({
                'Child Table': [tables[0]],
                'Parent Table': [tables[1]],
                'Metric': ['CardinalityShapeSimilariy'],
                'Quality Score': [scores['score']]
            })
            details_dataframe = pd.concat([details_dataframe, new_row], ignore_index=True)

        return details_dataframe

    def get_details(self, property_name, table_name=None):
        """Return the details for each score for the given property name.

        Args:
            property_name (str):
                The name of the property to return score details for.
            table_name (str):
                Optionally filter results by table.

        Returns:
            pd.DataFrame:
                The details of the scores of a property.
        """
        self._validate_inputs(property_name, table_name)

        property_instance = self._properties_instances[property_name]
        if property_name != 'Cardinality':
            return self._get_details_non_cardinality(property_instance, table_name)
        else:
            return self._get_details_cardinality(property_instance, table_name)

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
