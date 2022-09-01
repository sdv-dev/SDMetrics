"""Single table quality report."""

import itertools
import pickle
import sys

import numpy as np
import pandas as pd
import tqdm

from sdmetrics.reports.single_table.plot_utils import get_column_pairs_plot, get_column_shapes_plot
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
        self._real_corr = None
        self._synth_corr = None

    def _print_results(self, out=sys.stdout):
        """Print the quality report results."""
        out.write(f'Overall Quality Score: {self._overall_quality_score}\n\n')

        if len(self._property_breakdown) > 0:
            out.write('Properties:\n')

        for prop, score in self._property_breakdown.items():
            out.write(f'{prop}: {round(score * 100, 2)}%\n')

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
            self._metric_results[metric.__name__] = metric.compute_breakdown(
                real_data, synthetic_data)

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
        if property_name == 'Column Shapes':
            score_breakdowns = {
                metric.__name__: self._metric_results[metric.__name__]
                for metric in self.METRICS['Column Shapes']
            }
            fig = get_column_shapes_plot(score_breakdowns)

        elif property_name == 'Column Pairs':
            score_breakdowns = {
                metric.__name__: self._metric_results[metric.__name__]
                for metric in self.METRICS['Column Pair Trends']
            }
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

        for metric in self.METRICS[property_name]:
            for column, score_breakdown in self._metric_results[metric.__name__].items():
                columns.append(column)
                metrics.append(metric.__name__)
                scores.append(score_breakdown['score'])

        return pd.DataFrame({'Column': columns, 'Metric': metrics, 'Quality Score': scores})

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
