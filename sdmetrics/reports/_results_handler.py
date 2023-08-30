"""Class for handling the storage and displaying of results for reports."""

import sys

import pandas as pd


class BaseResultsHandler():
    """Base class for handling results for reports."""

    def print_results(self, report, verbose):
        """Print the results of a report.

        Args:
            report (sdmetrics.reports.BaseReport):
                Report class to print results for.
            verbose (bool):
                Whether or not to print results to std.out.
        """
        raise NotImplementedError


class QualityReportResultsHandler(BaseResultsHandler):
    """Results handler for quality reports."""

    def print_results(self, properties, score, verbose):
        """Print the results of a QualityReport.

        Args:
            properties (dict):
                Dictionary mapping property names to an instance of the Property class.
            score (float):
                Overall score of the report.
            verbose (bool):
                Whether or not to print results to std.out.
        """
        if verbose:
            sys.stdout.write(
                f'\nOverall Quality Score: {round(score * 100, 2)}%\n\n'
            )
            sys.stdout.write('Properties:\n')

            for property_name in properties:
                property_score = round(properties[property_name]._compute_average() * 100, 2)
                sys.stdout.write(
                    f'- {property_name}: {property_score}%\n'
                )


class DiagnosticReportResultsHandler(BaseResultsHandler):
    """Results handler for diagnostic reports."""

    DIAGNOSTIC_REPORT_RESULT_DETAILS = {
        'BoundaryAdherence': {
            'SUCCESS': (
                'The synthetic data follows over 90% of the min/max boundaries set by the real '
                'data'
            ),
            'WARNING': (
                'More than 10% the synthetic data does not follow the min/max boundaries set by '
                'the real data'
            ),
            'DANGER': (
                'More than 50% the synthetic data does not follow the min/max boundaries set by '
                'the real data'
            ),
        },
        'CategoryCoverage': {
            'SUCCESS': (
                'The synthetic data covers over 90% of the categories present in the real data'
            ),
            'WARNING': (
                'The synthetic data is missing more than 10% of the categories present in the '
                'real data'
            ),
            'DANGER': (
                'The synthetic data is missing more than 50% of the categories present in the '
                'real data'
            ),
        },
        'NewRowSynthesis': {
            'SUCCESS': 'Over 90% of the synthetic rows are not copies of the real data',
            'WARNING': 'More than 10% of the synthetic rows are copies of the real data',
            'DANGER': 'More than 50% of the synthetic rows are copies of the real data',
        },
        'RangeCoverage': {
            'SUCCESS': (
                'The synthetic data covers over 90% of the numerical ranges present in the real '
                'data'
            ),
            'WARNING': (
                'The synthetic data is missing more than 10% of the numerical ranges present in '
                'the real data'
            ),
            'DANGER': (
                'The synthetic data is missing more than 50% of the numerical ranges present in '
                'the real data'
            ),
        }
    }

    def __init__(self):
        self.results = {}

    def _print_results_for_level(self, level):
        """Print the result for a given level.

        Args:
            level (string):
                The level to print results for.
        """
        level_marks = {'SUCCESS': '✓', 'WARNING': '!', 'DANGER': 'x'}

        if len(self.results[level]) > 0:
            sys.stdout.write(f'\n{level}:\n')
            for result in self.results[level]:
                sys.stdout.write(f'{level_marks[level]} {result}\n')

    def print_results(self, properties, verbose):
        """Print the results of a DiagnosticReport.

        Args:
            properties (dict):
                Dictionary mapping property names to an instance of the Property class.
            verbose (bool):
                Whether or not to print results to std.out.
        """
        self.results['SUCCESS'] = []
        self.results['WARNING'] = []
        self.results['DANGER'] = []
        for property_name in properties:
            details = properties[property_name].details
            average_score_metric = details.groupby('Metric')['Score'].mean()
            for metric, score in average_score_metric.items():
                if pd.isna(score):
                    continue
                if score >= 0.9:
                    self.results['SUCCESS'].append(
                        self.DIAGNOSTIC_REPORT_RESULT_DETAILS[metric]['SUCCESS'])
                elif score >= 0.5:
                    self.results['WARNING'].append(
                        self.DIAGNOSTIC_REPORT_RESULT_DETAILS[metric]['WARNING'])
                else:
                    self.results['DANGER'].append(
                        self.DIAGNOSTIC_REPORT_RESULT_DETAILS[metric]['DANGER'])

        if verbose:
            sys.stdout.write('\nDiagnostic Results:\n')
            self._print_results_for_level('SUCCESS')
            self._print_results_for_level('WARNING')
            self._print_results_for_level('DANGER')
