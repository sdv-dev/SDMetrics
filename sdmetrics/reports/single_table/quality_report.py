"""Single table quality report."""

import itertools
import pickle
import sys
import warnings

import numpy as np
import pandas as pd
import pkg_resources
import tqdm

from sdmetrics.reports.single_table._properties import (
    ColumnPairTrends, ColumnShapes)
from sdmetrics.errors import IncomputableMetricError
from sdmetrics.reports.single_table.plot_utils import get_column_pairs_plot, get_column_shapes_plot
from sdmetrics.reports.utils import (
    aggregate_metric_results, discretize_and_apply_metric, validate_single_table_inputs)
from sdmetrics.single_table import (
    ContingencySimilarity, CorrelationSimilarity, KSComplement, TVComplement)


class QualityReport():
    """Single table quality report.

    This class creates a quality report for single-table data. It calculates the quality
    score along two properties - Column Shapes and Column Pair Trends.
    """

    METRICS = {
        'Column Shapes': [KSComplement, TVComplement],
        'Column Pair Trends': [CorrelationSimilarity, ContingencySimilarity],
    }

    def __init__(self):
        self._overall_quality_score = None
        self._metric_results = {}
        self._property_breakdown = {}
        self._property_errors = {}
        self.is_generated = False
        self._properties = {
            'Column Shapes': ColumnShapes(),
            'Column Pair Trends': ColumnPairTrends()
        }

    def _print_results(self, out=sys.stdout):
        """Print the quality report results."""
        if pd.isna(self._overall_quality_score) & any(self._property_errors.values()):
            out.write('\nOverall Quality Score: Error computing report.\n\n')
        else:
            out.write(
                f'\nOverall Quality Score: {round(self._overall_quality_score * 100, 2)}%\n\n')

        if len(self._property_breakdown) > 0:
            out.write('Properties:\n')

        for prop, score in self._property_breakdown.items():
            if not pd.isna(score):
                out.write(f'{prop}: {round(score * 100, 2)}%\n')
            elif self._property_errors[prop] > 0:
                out.write(f'{prop}: Error computing property.\n')
            else:
                out.write(f'{prop}: NaN\n')

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
        validate_single_table_inputs(real_data, synthetic_data, metadata)

        scores = []
        num_columns = len(metadata['columns'])
        for property_name in self._properties:
            if property_name == 'Column Shapes':
                num_iterations = num_columns
            elif property_name == 'Column Pair Trends':
                num_iterations = int(0.5 * num_columns * num_columns - 1)
            
            if verbose:
                out = tqdm.tqdm(total=num_iterations, file=sys.stdout)
                out.set_description(f'Computing {property_name}...')

            scores.append(self._properties[property_name].get_score(
                real_data, synthetic_data, metadata, progress_bar=out)
            )

        self._overall_quality_score = np.nanmean(scores)

        if verbose:
            self._print_results()

    def _validate_property_generation(self, property_name):
        """Validate that the given property name is valid and that the report has been generated."""
        if property_name not in ['ColumnShapes', 'ColumnPairTrends']:
            raise ValueError(
                f"Invalid property name '{property_name}'."
                "Valid property names are 'ColumnShapes' and 'ColumnPairTrends'."
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
        """Return the property score breakdown.

        Returns:
            pandas.DataFrame
                The property score breakdown.
        """
        return pd.DataFrame({
            'Property': self._property_breakdown.keys(),
            'Score': self._property_breakdown.values(),
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
        self._validate_property_generation(property_name)

        return self._properties[property_name].get_visualization()

    def get_details(self, property_name):
        """Return the details table for the given property name.

        Args:
            property_name (str):
                The name of the property to return details for.

        Returns:
            pandas.DataFrame
        """
        self._validate_property_generation(property_name)
        
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
