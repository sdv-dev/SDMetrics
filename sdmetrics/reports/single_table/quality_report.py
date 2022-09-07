"""Single table quality report."""

import copy
import itertools
import pickle
import sys

import numpy as np
import pandas as pd
import tqdm

from sdmetrics.reports.single_table.plot_utils import get_column_pairs_plot, get_column_shapes_plot
from sdmetrics.single_table import (
    ContingencySimilarity, CorrelationSimilarity, KSComplement, TVComplement)
from sdmetrics.utils import is_datetime


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
        self._real_corr = None
        self._synth_corr = None

    def _print_results(self, out=sys.stdout):
        """Print the quality report results."""
        out.write(f'Overall Quality Score: {self._overall_quality_score}\n\n')

        if len(self._property_breakdown) > 0:
            out.write('Properties:\n')

        for prop, score in self._property_breakdown.items():
            out.write(f'{prop}: {round(score * 100, 2)}%\n')

    def _discretize_data(self, real_data, synthetic_data, metadata):
        """Create a copy of the real and synthetic data with discretized data.

        Convert numerical and datetime columns to discrete values, and label them
        as categorical.

        Args:
            real_data (pandas.DataFrame):
                The real data.
            synthetic_data (pandas.DataFrame):
                The synthetic data.
            metadata (dict)
                The metadata.

        Returns:
            (pandas.DataFrame, pandas.DataFrame, dict):
                The binned real and synthetic data, and the updated metadata.
        """
        binned_real = real_data.copy()
        binned_synthetic = synthetic_data.copy()
        binned_metadata = copy.deepcopy(metadata)

        for field_name, field_meta in metadata['fields'].items():
            if field_meta['type'] == 'numerical' or field_meta['type'] == 'datetime':
                real_col = real_data[field_name]
                synthetic_col = synthetic_data[field_name]
                if is_datetime(real_col):
                    real_col = pd.to_numeric(real_col)
                    synthetic_col = pd.to_numeric(synthetic_col)

                bin_edges = np.histogram_bin_edges(real_col)
                binned_real_col = np.digitize(real_col, bins=bin_edges)
                binned_synthetic_col = np.digitize(synthetic_col, bins=bin_edges)

                binned_real[field_name] = binned_real_col
                binned_synthetic[field_name] = binned_synthetic_col
                binned_metadata['fields'][field_name] = {'type': 'categorical'}

        return binned_real, binned_synthetic, binned_metadata

    def generate(self, real_data, synthetic_data, metadata):
        """Generate report.

        Args:
            real_data (pandas.DataFrame):
                The real data.
            synthetic_data (pandas.DataFrame):
                The synthetic data.
            metadata (dict):
                The metadata, which contains each column's data type as well as relationships.
        """
        metrics = list(itertools.chain.from_iterable(self.METRICS.values()))

        for metric in tqdm.tqdm(metrics, desc='Creating report:'):
            if metric in self.METRICS['Column Pair Trends']:
                binned_real, binned_synthetic, binned_metadata = self._discretize_data(
                    real_data, synthetic_data, metadata)
                self._metric_results[metric.__name__] = metric.compute_breakdown(
                    binned_real, binned_synthetic, binned_metadata)
            else:
                self._metric_results[metric.__name__] = metric.compute_breakdown(
                    real_data, synthetic_data, metadata)

        self._property_breakdown = {}
        for prop, metrics in self.METRICS.items():
            prop_scores = []
            for metric in metrics:
                score = np.nanmean(
                    [
                        breakdown['score'] for _, breakdown
                        in self._metric_results[metric.__name__].items()
                    ]
                )
                prop_scores.append(score)

            self._property_breakdown[prop] = np.mean(prop_scores)

        # Calculate and store the correlation matrices.
        self._real_corr = real_data.dropna().corr()
        self._synth_corr = synthetic_data.dropna().corr()

        self._overall_quality_score = np.mean(list(self._property_breakdown.values()))

        self._print_results()

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

    def show_details(self, property_name):
        """Display a visualization for each score for the given property name.

        Args:
            property_name (str):
                The name of the property to return score details for.
        """
        score_breakdowns = {
            metric.__name__: self._metric_results[metric.__name__]
            for metric in self.METRICS.get(property_name, [])
        }

        if property_name == 'Column Shapes':
            fig = get_column_shapes_plot(score_breakdowns)

        elif property_name == 'Column Pair Trends':
            fig = get_column_pairs_plot(score_breakdowns, self._real_corr, self._synth_corr)

        fig.show()

    def get_details(self, property_name):
        """Return the details for each score for the given property name.

        Args:
            property_name (str):
                The name of the property to return score details for.

        Returns:
            pandas.DataFrame
                The score breakdown.
        """
        columns = []
        metrics = []
        scores = []

        if property_name == 'Column Shapes':
            for metric in self.METRICS[property_name]:
                for column, score_breakdown in self._metric_results[metric.__name__].items():
                    columns.append(column)
                    metrics.append(metric.__name__)
                    scores.append(score_breakdown['score'])

            return pd.DataFrame({
                'Column': columns,
                'Metric': metrics,
                'Quality Score': scores,
            })

        elif property_name == 'Column Pair Trends':
            real_scores = []
            synthetic_scores = []
            for metric in self.METRICS[property_name]:
                for column_pair, score_breakdown in self._metric_results[metric.__name__].items():
                    columns.append(column_pair)
                    metrics.append(metric.__name__)
                    scores.append(score_breakdown['score'])
                    real_scores.append(score_breakdown['real'])
                    synthetic_scores.append(score_breakdown['synthetic'])

            return pd.DataFrame({
                'Columns': columns,
                'Metric': metrics,
                'Quality Score': scores,
                'Real Score': real_scores,
                'Synthetic Score': synthetic_scores,
            })

    def get_raw_result(self, metric_name):
        """Return the raw result of the given metric name.

        Args:
            metric_name (str):
                The name of the desired metric.

        Returns:
            dict
                The raw results
        """
        metrics = list(itertools.chain.from_iterable(self.METRICS.values()))
        for metric in metrics:
            if metric.__name__ == metric_name:
                return {
                    'metric': f'{metric.__module__}.{metric.__name__}',
                    'results': self._metric_results[metric_name],
                }

    def save(self, filename):
        """Save this report instance to the given path using pickle.

        Args:
            filename (str):
                File where the report instance will be serialized.
        """
        with open(filename, 'wb') as output:
            pickle.dump(self, output)

    @classmethod
    def load(cls, filename):
        """Load a ``QualityReport`` instance from a given path.

        Args:
            filename (str):
                File from which to load the instance.

        Returns:
            QualityReort:
                The loaded quality report instance.
        """
        with open(filename, 'rb') as f:
            return pickle.load(f)
