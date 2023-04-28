"""Single table diagnostic report."""

import copy
import itertools
import logging
import pickle
import sys
import warnings

import numpy as np
import pandas as pd
import pkg_resources
import tqdm

from sdmetrics.errors import IncomputableMetricError
from sdmetrics.reports.single_table.plot_utils import (
    get_column_boundaries_plot, get_column_coverage_plot, get_synthesis_plot)
from sdmetrics.reports.utils import (
    DIAGNOSTIC_REPORT_RESULT_DETAILS, aggregate_metric_results, print_results_for_level,
    validate_single_table_inputs)
from sdmetrics.single_table import (
    BoundaryAdherence, CategoryCoverage, NewRowSynthesis, RangeCoverage)

LOGGER = logging.getLogger(__name__)


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
    _METRIC_ARGS = {
        'NewRowSynthesis': {'synthetic_sample_size': 10000},
    }

    def __init__(self):
        self._metric_results = {}
        self._metric_averages = {}
        self._property_scores = {}
        self._results = {}
        self._metric_args = copy.deepcopy(self._METRIC_ARGS)

    def _print_results(self, out=sys.stdout):
        """Print the diagnostic report results."""
        self._results['SUCCESS'] = []
        self._results['WARNING'] = []
        self._results['DANGER'] = []

        for metric, score in self._metric_averages.items():
            if pd.isna(score):
                continue
            if score >= 0.9:
                self._results['SUCCESS'].append(
                    DIAGNOSTIC_REPORT_RESULT_DETAILS[metric]['SUCCESS'])
            elif score >= 0.5:
                self._results['WARNING'].append(
                    DIAGNOSTIC_REPORT_RESULT_DETAILS[metric]['WARNING'])
            else:
                self._results['DANGER'].append(
                    DIAGNOSTIC_REPORT_RESULT_DETAILS[metric]['DANGER'])

        out.write('\nDiagnosticResults:\n')
        print_results_for_level(out, self._results, 'SUCCESS')
        print_results_for_level(out, self._results, 'WARNING')
        print_results_for_level(out, self._results, 'DANGER')

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

        metrics = list(itertools.chain.from_iterable(self.METRICS.values()))
        self._metric_args['NewRowSynthesis']['synthetic_sample_size'] = min(
            min(len(real_data), len(synthetic_data)),
            self._metric_args['NewRowSynthesis']['synthetic_sample_size'],
        )

        for metric in tqdm.tqdm(metrics, desc='Creating report', disable=(not verbose)):
            metric_name = metric.__name__
            try:
                metric_args = self._metric_args.get(metric_name, {})
                self._metric_results[metric_name] = metric.compute_breakdown(
                    real_data, synthetic_data, metadata, **metric_args)

                if 'score' in self._metric_results[metric_name]:
                    self._metric_averages[metric_name] = self._metric_results[metric_name]['score']
                else:
                    avg_score, _ = aggregate_metric_results(self._metric_results[metric_name])
                    self._metric_averages[metric_name] = avg_score

            except Exception as e:
                # Metric is not compatible with this dataset.
                self._metric_results[metric_name] = {}
                self._metric_averages[metric_name] = np.nan
                if not isinstance(e, IncomputableMetricError):
                    msg = f'Unexpected error occured when calculating {metric_name} metric:'
                    LOGGER.error(msg, exc_info=1)

        self._property_scores = {}
        for prop, _ in self.METRICS.items():
            self._property_scores[prop] = self.get_details(prop)['Diagnostic Score'].mean()

        if verbose:
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
        return copy.deepcopy(self._property_scores)

    def get_visualization(self, property_name):
        """Return a visualization for each score for the given property name.

        Args:
            property_name (str):
                The name of the property to return score details for.

        Returns:
            plotly.graph_objects._figure.Figure
                The visualization for the requested property.
        """
        score_breakdowns = {
            metric.__name__: self._metric_results[metric.__name__]
            for metric in self.METRICS.get(property_name, [])
        }

        if property_name == 'Coverage':
            fig = get_column_coverage_plot(score_breakdowns, self._property_scores[property_name])

        elif property_name == 'Boundaries':
            fig = get_column_boundaries_plot(
                score_breakdowns, self._property_scores[property_name])

        elif property_name == 'Synthesis':
            fig = get_synthesis_plot(score_breakdowns.get('NewRowSynthesis', {}))

        else:
            raise ValueError(f'Property name `{property_name}` is not recognized. '
                             'Please choose either `Coverage`, `Boundaries`, or `Synthesis`.')

        return fig

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
            metric_result = self._metric_results[metric_name]
            details = pd.DataFrame({
                'Metric': [metric_name],
                'Diagnostic Score': [metric_result.get('score', np.nan)],
                'Num Matched Rows': [metric_result.get('num_matched_rows', np.nan)],
                'Num New Rows': [metric_result.get('num_new_rows', np.nan)],
            })
            errors.append(metric_result.get('error', np.nan))

        else:
            for metric in self.METRICS[property_name]:
                for column, score_breakdown in self._metric_results[metric.__name__].items():
                    metric_score = score_breakdown.get('score', np.nan)
                    metric_error = score_breakdown.get('error', np.nan)
                    if pd.isna(metric_score) and pd.isna(metric_error):
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
