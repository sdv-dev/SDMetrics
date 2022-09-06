"""Multi table quality report."""

import itertools
import pickle
import sys

import numpy as np
import pandas as pd
import tqdm

from sdmetrics.multi_table import (
    CardinalityShapeSimilarity, ContingencySimilarity, CorrelationSimilarity, KSComplement,
    TVComplement)
from sdmetrics.reports.multi_table.plot_utils import get_table_relationships_plot
from sdmetrics.reports.single_table.plot_utils import get_column_pairs_plot, get_column_shapes_plot


class QualityReport():
    """Multi table quality report.

    This class creates a quality report for multi-table data. It calculates the quality
    score along three properties - Column Shapes, Column Pair Trends, and Parent Child
    Relationships.
    """

    METRICS = {
        'Column Shapes': [KSComplement, TVComplement],
        'Column Pair Trends': [CorrelationSimilarity, ContingencySimilarity],
        'Parent Child Relationships': [CardinalityShapeSimilarity],
    }

    def __init__(self):
        self._overall_quality_score = None
        self._metric_results = {}
        self._property_breakdown = {}

    def _print_results(self, out=sys.stdout):
        """Print the quality report results."""
        out.write(f'Overall Quality Score: {self._overall_quality_score}\n')

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
                real_data, synthetic_data, metadata)

        self._property_breakdown = {}
        for prop, metrics in self.METRICS.items():
            prop_scores = []
            if prop == 'Parent Child Relationships':
                for metric in metrics:
                    score = np.nanmean(
                        [
                            table_breakdown['score'] for _, table_breakdown
                            in self._metric_results['CardinalityShapeSimilarity'].items()
                        ]
                    )
                    prop_scores.append(score)
            else:
                for metric in metrics:
                    score = np.nanmean(
                        [
                            breakdown['score'] for _, table_breakdowns
                            in self._metric_results[metric.__name__].items()
                            for _, breakdown in table_breakdowns.items()
                        ]
                    )
                    prop_scores.append(score)

            self._property_breakdown[prop] = np.mean(prop_scores)

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

    def show_details(self, property_name, table_name=None):
        """Display a visualization for each score for the given property and table.

        Args:
            property_name (str):
                The name of the property to return score details for.
            table_name (str):
                The table to show scores for. Must be provided for 'Column Shapes'
                and 'Column Pair Trends'
        """
        if property_name in ['Column Shapes', 'Column Pair Trends'] and table_name is None:
            raise ValueError('Table name must be provided when viewing details for '
                             f'property {property_name}.')

        if property_name == 'Column Shapes':
            score_breakdowns = {
                metric.__name__: self._metric_results[metric.__name__].get(table_name, {})
                for metric in self.METRICS.get(property_name, [])
            }
            fig = get_column_shapes_plot(score_breakdowns)

        elif property_name == 'Column Pair Trends':
            score_breakdowns = {
                metric.__name__: self._metric_results[metric.__name__].get(table_name, {})
                for metric in self.METRICS.get(property_name, [])
            }
            fig = get_column_pairs_plot(score_breakdowns, self._real_corr, self._synth_corr)

        elif property_name == 'Parent Child Relationships':
            score_breakdowns = {
                metric.__name__: self._metric_results[metric.__name__]
                for metric in self.METRICS.get(property_name, [])
            }
            fig = get_table_relationships_plot(score_breakdowns)

        fig.show()

    def get_details(self, property_name, table_name=None):
        """Return the details for each score for the given property name.

        Args:
            property_name (str):
                The name of the property to return score details for.
            table_name (str):
                Optionally filter results by table.

        Returns:
            pandas.DataFrame
                The score breakdown.
        """
        tables = []
        columns = []
        metrics = []
        scores = []

        if property_name == 'Column Shapes':
            for metric in self.METRICS[property_name]:
                for table, table_breakdown in self._metric_results[metric.__name__].items():
                    if table_name is not None and table != table_name:
                        continue

                    for column, score_breakdown in table_breakdown.items():
                        tables.append(table)
                        columns.append(column)
                        metrics.append(metric.__name__)
                        scores.append(score_breakdown['score'])

            return pd.DataFrame({
                'Table Name': tables,
                'Column': columns,
                'Metric': metrics,
                'Quality Score': scores,
            })

        elif property_name == 'Column Pair Trends':
            real_scores = []
            synthetic_scores = []
            for metric in self.METRICS[property_name]:
                for table, table_breakdown in self._metric_results[metric.__name__].items():
                    if table_name is not None and table != table_name:
                        continue

                    for column_pair, score_breakdown in table_breakdown.items():
                        tables.append(table)
                        columns.append(column_pair)
                        metrics.append(metric.__name__)
                        scores.append(score_breakdown['score'])
                        real_scores.append(score_breakdown['real'])
                        synthetic_scores.append(score_breakdown['synthetic'])

            return pd.DataFrame({
                'Table Name': tables,
                'Columns': columns,
                'Metric': metrics,
                'Quality Score': scores,
                'Real Score': real_scores,
                'Synthetic Score': synthetic_scores,
            })

        elif property_name == 'Parent Child Relationships':
            child_tables = []
            for metric in self.METRICS[property_name]:
                for tables, score_breakdown in self._metric_results[metric.__name__].items():
                    tables.append(tables[0])
                    child_tables.append(tables[1])
                    metrics.append(metric.__name__)
                    scores.append(score_breakdown['score'])

            return pd.DataFrame({
                'Child Table': child_tables,
                'Parent Table': tables,
                'Metric': metrics,
                'Quality Score': scores,
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
