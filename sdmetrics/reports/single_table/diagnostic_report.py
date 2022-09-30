"""Single table diagnostic report."""

import copy
import itertools
import pickle
import sys
import warnings

import numpy as np
import pandas as pd
import pkg_resources
import tqdm

from sdmetrics.errors import IncomputableMetricError
from sdmetrics.reports.utils import aggregate_metric_results
from sdmetrics.single_table import (
    BoundaryAdherence, CategoryCoverage, NewRowSynthesis, RangeCoverage)

RESULT_DETAILS = {
    'BoundaryAdherence': {
        'SUCCESS': (
            '✓ The synthetic data general follows the min/max boundaries set by the real data'
        ),
        'WARNING': (
            '! More than 10% the synthetic data does not follow the min/max boundaries set by '
            'the real data'
        ),
        'DANGER': (
            'x More than 50% the synthetic data does not follow the min/max boundaries set by '
            'the real data'
        ),
    },
    'CategoryCoverage': {
        'SUCCESS': '✓ The synthetic data generally covers categories present in the real data',
        'WARNING': (
            '! The synthetic data is missing more than 10% of the categories present in the '
            'real data'
        ),
        'DANGER': (
            'x The synthetic data is missing more than 50% of the categories present in the '
            'real data'
        ),
    },
    'NewRowSynthesis': {
        'SUCCESS': '✓ The synthetic rows are generally not copies of the real data',
        'WARNING': '! More than 10% of the synthetic rows are copies of the real data',
        'DANGER': 'x More than 50% of the synthetic rows are copies of the real data',
    },
    'RangeCoverage': {
        'SUCCESS': (
            '✓ The synthetic data generally covers numerical ranges present in the real data'
        ),
        'WARNING': (
            '! The synthetic data is missing more than 10% of the numerical ranges present in '
            'the real data'
        ),
        'DANGER': (
            'x The synthetic data is missing more than 50% of the numerical ranges present in '
            'the real data'
        ),
    }
}


class DiagnosticReport():
    """Single table diagnostic report.

    This class creates a diagnostic report for single-table data. It calculates the diagnostic
    score along three properties - synthesis, coverage, and boundaries.
    """

    METRICS = {
        'Coverage': [RangeCoverage, CategoryCoverage],
        'Synthesis': [NewRowSynthesis],
        'Boundaries': [BoundaryAdherence],
    }

    def __init__(self):
        self._metric_results = {}
        self._metric_averages = {}
        self._results = {}

    def _print_results_for_level(self, out, level):
        """Print the result for a given level.

        Args:
            level (string):
                The level to print results for.
        """
        if len(self._results[level]) > 0:
            out.write(f'\n{level}:\n')
            for result in self._results[level]:
                out.write(f'{result}\n')

    def _print_results(self, out=sys.stdout):
        """Print the diagnostic report results."""
        self._results['SUCCESS'] = []
        self._results['WARNING'] = []
        self._results['DANGER'] = []

        for metric, score in self._metric_averages.items():
            if np.isnan(score):
                continue
            if score >= 0.9:
                self._results['SUCCESS'].append(RESULT_DETAILS[metric]['SUCCESS'])
            elif score >= 0.5:
                self._results['WARNING'].append(RESULT_DETAILS[metric]['WARNING'])
            else:
                self._results['DANGER'].append(RESULT_DETAILS[metric]['DANGER'])

        out.write('DiagnosticResults:\n')
        self._print_results_for_level(out, 'SUCCESS')
        self._print_results_for_level(out, 'WARNING')
        self._print_results_for_level(out, 'DANGER')

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

        for metric in tqdm.tqdm(metrics, desc='Creating report'):
            metric_name = metric.__name__
            try:
                self._metric_results[metric_name] = metric.compute_breakdown(
                    real_data, synthetic_data, metadata)

                if 'score' in self._metric_results[metric_name]:
                    self._metric_averages[metric_name] = self._metric_results[metric_name]['score']
                else:
                    avg_score, _ = aggregate_metric_results(self._metric_results[metric_name])
                    self._metric_averages[metric_name] = avg_score

            except IncomputableMetricError:
                # Metric is not compatible with this dataset.
                self._metric_results[metric_name] = {}

        self._print_results()

    def get_results(self):
        """Return the diagnostic results.

        Returns:
            dict
                The diagnostic results.
        """
        return copy.deepcopy(self._results)

    def get_properties(self):
        """Return the property score breakdown.

        Returns:
            pandas.DataFrame
                The property score breakdown.
        """
        properties = {}
        for prop, metrics in self.METRICS.items():
            prop_scores = [self._metric_averages[metric.__name__] for metric in metrics]
            properties[prop] = np.nanmean(prop_scores)

        return properties

    def get_visualization(self, property_name):
        """Return a visualization for each score for the given property name.

        Args:
            property_name (str):
                The name of the property to return score details for.

        Returns:
            plotly.graph_objects._figure.Figure
                The visualization for the requested property.
        """

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
        errors = []
        details = pd.DataFrame()

        if property_name == 'Synthesis':
            metric_name = self.METRICS[property_name][0].__name__
            details = pd.DataFrame({
                'Metric': [metric_name],
                'Diagnostic Score': [self._metric_results[metric_name].get('score', np.nan)],
            })
            errors.append(self._metric_results[metric_name].get('error', np.nan))

        else:
            for metric in self.METRICS[property_name]:
                for column, score_breakdown in self._metric_results[metric.__name__].items():
                    metric_score = score_breakdown.get('score', np.nan)
                    metric_error = score_breakdown.get('error', np.nan)
                    if np.isnan(metric_score) and np.isnan(metric_error):
                        continue

                    columns.append(column)
                    metrics.append(metric.__name__)
                    scores.append(metric_score)
                    errors.append(metric_error)

            details = pd.DataFrame({
                'Column': columns,
                'Metric': metrics,
                'Diagnostic Score': scores,
            })

        if pd.Series(errors).notna().sum() > 0:
            details['Error'] = errors

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
                The loaded diagnostic report instance.
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
