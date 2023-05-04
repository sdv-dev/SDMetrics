"""Multi table diagnostic report."""

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
from sdmetrics.multi_table import (
    BoundaryAdherence, CategoryCoverage, NewRowSynthesis, RangeCoverage)
from sdmetrics.reports.single_table.plot_utils import (
    get_column_boundaries_plot, get_column_coverage_plot, get_synthesis_plot)
from sdmetrics.reports.utils import (
    DIAGNOSTIC_REPORT_RESULT_DETAILS, aggregate_metric_results, print_results_for_level,
    validate_multi_table_inputs)

LOGGER = logging.getLogger(__name__)


class DiagnosticReport():
    """Multi table diagnostic report.

    This class creates a diagnostic report for multi-table data. It calculates the diagnostic
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
        self._metric_averages_by_table = {}
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
        validate_multi_table_inputs(real_data, synthetic_data, metadata)

        metrics = list(itertools.chain.from_iterable(self.METRICS.values()))
        for metric in tqdm.tqdm(metrics, desc='Creating report', disable=(not verbose)):
            metric_name = metric.__name__
            try:
                metric_args = self._metric_args.get(metric_name, {})
                self._metric_results[metric_name] = metric.compute_breakdown(
                    real_data, synthetic_data, metadata, **metric_args)
                self._metric_averages_by_table[metric_name] = {}

                metric_scores = []
                for table_name, table_breakdown in self._metric_results[metric_name].items():
                    if 'score' in table_breakdown:
                        avg_table_score = table_breakdown['score']
                    else:
                        avg_table_score, _ = aggregate_metric_results(table_breakdown)

                    self._metric_averages_by_table[metric_name][table_name] = avg_table_score
                    if not pd.isna(avg_table_score):
                        metric_scores.append(avg_table_score)

                self._metric_averages[metric_name] = np.mean(metric_scores) if (
                    len(metric_scores) > 0) else np.nan

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

    def get_visualization(self, property_name, table_name):
        """Return a visualization for each score for the given property name.

        Args:
            property_name (str):
                The name of the property to generate a visualization for.
            table_name (str):
                The table to generate a visualization for.

        Returns:
            plotly.graph_objects._figure.Figure
                The visualization for the requested property.
        """
        score_breakdowns = {}
        table_scores = []
        for metric in self.METRICS.get(property_name, []):
            score_breakdowns[metric.__name__] = self._metric_results[metric.__name__].get(
                table_name, {})

            metric_table_score = self._metric_averages_by_table[metric.__name__].get(
                table_name, np.nan)
            if not pd.isna(metric_table_score):
                table_scores.append(metric_table_score)

        average_table_score = np.nan
        if len(table_scores) > 0:
            average_table_score = np.mean(table_scores)

        if property_name == 'Coverage':
            fig = get_column_coverage_plot(score_breakdowns, average_table_score)

        elif property_name == 'Boundaries':
            fig = get_column_boundaries_plot(score_breakdowns, average_table_score)

        elif property_name == 'Synthesis':
            fig = get_synthesis_plot(score_breakdowns.get('NewRowSynthesis', {}))

        else:
            raise ValueError(f'Property name `{property_name}` is not recognized. '
                             'Please choose either `Coverage`, `Boundaries`, or `Synthesis`.')

        return fig

    def get_details(self, property_name, table_name=None):
        """Return the details for each score for the given property name.

        Args:
            property_name (str):
                The name of the property to return score details for.
            table_name (str):
                If provided, filter the details by the given table.

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

        if property_name == 'Synthesis':
            matched_rows = []
            new_rows = []
            metric_name = self.METRICS[property_name][0].__name__
            for table, table_breakdown in self._metric_results[metric_name].items():
                if table_name is not None and table != table_name:
                    continue

                tables.append(table)
                metrics.append(metric_name)
                scores.append(table_breakdown.get('score', np.nan))
                matched_rows.append(table_breakdown.get('num_matched_rows', np.nan))
                new_rows.append(table_breakdown.get('num_new_rows', np.nan))
                errors.append(table_breakdown.get('error', np.nan))

            details = pd.DataFrame({
                'Table': tables,
                'Metric': metrics,
                'Diagnostic Score': scores,
                'Num Matched Rows': matched_rows,
                'Num New Rows': new_rows,
            })

        else:
            for metric in self.METRICS[property_name]:
                for table, table_breakdown in self._metric_results[metric.__name__].items():
                    if table_name is not None and table != table_name:
                        continue

                    for column, score_breakdown in table_breakdown.items():
                        metric_score = score_breakdown.get('score', np.nan)
                        metric_error = score_breakdown.get('error', np.nan)
                        if pd.isna(metric_score) and pd.isna(metric_error):
                            continue

                        tables.append(table)
                        columns.append(column)
                        metrics.append(metric.__name__)
                        scores.append(metric_score)
                        errors.append(metric_error)

            details = pd.DataFrame({
                'Table': tables,
                'Column': columns,
                'Metric': metrics,
                'Diagnostic Score': scores,
            })

        if pd.Series(errors).notna().sum() > 0:
            details['Error'] = errors

        return details.sort_values(by=['Table'], ignore_index=True)

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
