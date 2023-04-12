"""Multi table quality report."""

import itertools
import pickle
import sys
import warnings

import numpy as np
import pandas as pd
import pkg_resources
import tqdm

from sdmetrics.errors import IncomputableMetricError
from sdmetrics.multi_table import (
    CardinalityShapeSimilarity, ContingencySimilarity, CorrelationSimilarity, KSComplement,
    TVComplement)
from sdmetrics.reports.multi_table.plot_utils import get_table_relationships_plot
from sdmetrics.reports.single_table.plot_utils import get_column_pairs_plot, get_column_shapes_plot
from sdmetrics.reports.utils import (
    aggregate_metric_results, discretize_and_apply_metric, validate_multi_table_inputs)


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
        self._property_errors = {}

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
        validate_multi_table_inputs(real_data, synthetic_data, metadata)

        metrics = list(itertools.chain.from_iterable(self.METRICS.values()))

        for metric in tqdm.tqdm(metrics, desc='Creating report', disable=(not verbose)):
            try:
                self._metric_results[metric.__name__] = metric.compute_breakdown(
                    real_data, synthetic_data, metadata)
            except IncomputableMetricError:
                # Metric is not compatible with this dataset.
                self._metric_results[metric.__name__] = {}
                continue

        for table_name, table_data in real_data.items():
            existing_column_pairs = []
            if table_name in self._metric_results['ContingencySimilarity']:
                existing_column_pairs.append(
                    list(self._metric_results['ContingencySimilarity'][table_name].keys()))
            else:
                self._metric_results['ContingencySimilarity'][table_name] = {}

            if table_name in self._metric_results['CorrelationSimilarity']:
                existing_column_pairs.append(
                    list(self._metric_results['CorrelationSimilarity'][table_name].keys()))

            additional_results = discretize_and_apply_metric(
                real_data[table_name],
                synthetic_data[table_name],
                metadata['tables'][table_name],
                ContingencySimilarity,
                existing_column_pairs,
            )
            self._metric_results['ContingencySimilarity'][table_name].update(additional_results)

        self._property_breakdown = {}
        for prop, metrics in self.METRICS.items():
            prop_scores = []
            num_prop_errors = 0
            if prop == 'Parent Child Relationships':
                for metric in metrics:
                    if 'score' in self._metric_results[metric.__name__]:
                        prop_scores.append(self._metric_results[metric.__name__]['score'])
                    else:
                        _, num_metric_errors = aggregate_metric_results(
                            self._metric_results[metric.__name__])
                        num_prop_errors += num_metric_errors
            else:
                for metric in metrics:
                    for _, table_breakdowns in self._metric_results[metric.__name__].items():
                        _, num_metric_errors = aggregate_metric_results(table_breakdowns)
                        num_prop_errors += num_metric_errors

            self._property_breakdown[prop] = np.nanmean(prop_scores) if (
                len(prop_scores) > 0
            ) else self.get_details(prop)['Quality Score'].mean()
            self._property_errors[prop] = num_prop_errors

        self._overall_quality_score = np.nanmean(list(self._property_breakdown.values()))

        if verbose:
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

    def get_visualization(self, property_name, table_name=None):
        """Return a visualization for each score for the given property and table.

        Args:
            property_name (str):
                The name of the property to return score details for.
            table_name (str):
                The table to show scores for. Must be provided for 'Column Shapes'
                and 'Column Pair Trends'

        Returns:
            plotly.graph_objects._figure.Figure
                A visualization of the requested property's scores.
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
            fig = get_column_pairs_plot(
                score_breakdowns,
            )

        elif property_name == 'Parent Child Relationships':
            score_breakdowns = {
                metric.__name__: self._metric_results[metric.__name__]
                for metric in self.METRICS.get(property_name, [])
            }
            if table_name is not None:
                for metric, metric_results in score_breakdowns.items():
                    score_breakdowns[metric] = {
                        tables: results for tables, results in metric_results.items()
                        if table_name in tables
                    }

            fig = get_table_relationships_plot(score_breakdowns)

        return fig

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
        errors = []
        details = pd.DataFrame()

        if property_name == 'Column Shapes':
            for metric in self.METRICS[property_name]:
                for table, table_breakdown in self._metric_results[metric.__name__].items():
                    if table_name is not None and table != table_name:
                        continue

                    for column, score_breakdown in table_breakdown.items():
                        if 'score' in score_breakdown and pd.isna(score_breakdown['score']):
                            continue
                        tables.append(table)
                        columns.append(column)
                        metrics.append(metric.__name__)
                        scores.append(score_breakdown.get('score', np.nan))
                        errors.append(score_breakdown.get('error', np.nan))

            details = pd.DataFrame({
                'Table': tables,
                'Column': columns,
                'Metric': metrics,
                'Quality Score': scores,
            }).sort_values(by=['Table'], ignore_index=True)

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
                        scores.append(score_breakdown.get('score', np.nan))
                        real_scores.append(score_breakdown.get('real', np.nan))
                        synthetic_scores.append(score_breakdown.get('synthetic', np.nan))
                        errors.append(score_breakdown.get('error', np.nan))

            details = pd.DataFrame({
                'Table': tables,
                'Column 1': [col1 for col1, _ in columns],
                'Column 2': [col2 for _, col2 in columns],
                'Metric': metrics,
                'Quality Score': scores,
                'Real Correlation': real_scores,
                'Synthetic Correlation': synthetic_scores,
            }).sort_values(by=['Table'], ignore_index=True)

        elif property_name == 'Parent Child Relationships':
            child_tables = []
            for metric in self.METRICS[property_name]:
                for table_pair, score_breakdown in self._metric_results[metric.__name__].items():
                    if table_name is not None and table_name not in table_pair:
                        continue

                    tables.append(table_pair[0])
                    child_tables.append(table_pair[1])
                    metrics.append(metric.__name__)
                    scores.append(score_breakdown.get('score', np.nan))
                    errors.append(score_breakdown.get('error', np.nan))

            details = pd.DataFrame({
                'Child Table': child_tables,
                'Parent Table': tables,
                'Metric': metrics,
                'Quality Score': scores,
            })

        if pd.Series(errors).notna().sum() > 0:
            details['Error'] = errors

        return details

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
                filtered_results = {}
                for table_name, table_results in self._metric_results[metric_name].items():
                    filtered_results[table_name] = {
                        key: result for key, result in table_results.items()
                        if not pd.isna(result['score'])
                    }

                return [
                    {
                        'metric': {
                            'method': f'{metric.__module__}.{metric.__name__}',
                            'parameters': {},
                        },
                        'results': filtered_results,
                    },
                ]

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
