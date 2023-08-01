"""Single table diagnostic report."""

import logging
import sys
from copy import deepcopy

import pandas as pd

from sdmetrics.reports.single_table._properties import Boundary, Coverage, Synthesis
from sdmetrics.reports.single_table.base_report import BaseReport
from sdmetrics.reports.utils import DIAGNOSTIC_REPORT_RESULT_DETAILS, print_results_for_level

LOGGER = logging.getLogger(__name__)


class DiagnosticReport(BaseReport):
    """Single table diagnostic report.

    This class creates a diagnostic report for single-table data. It calculates the diagnostic
    score along three properties - Synthesis, Coverage, and Boundary.
    """

    def __init__(self):
        super().__init__()
        self._properties = {
            'Coverage': Coverage(),
            'Boundary': Boundary(),
            'Synthesis': Synthesis()
        }
        self.results = {}

    def _generate_results(self):
        """Generate the diagnostic report results."""
        if not self.results:
            self.results['SUCCESS'] = []
            self.results['WARNING'] = []
            self.results['DANGER'] = []

            for property_name in self._properties:
                details = self._properties[property_name]._details
                average_score_metric = details.groupby('Metric')['Score'].mean()
                for metric, score in average_score_metric.items():
                    if pd.isna(score):
                        continue
                    if score >= 0.9:
                        self.results['SUCCESS'].append(
                            DIAGNOSTIC_REPORT_RESULT_DETAILS[metric]['SUCCESS'])
                    elif score >= 0.5:
                        self.results['WARNING'].append(
                            DIAGNOSTIC_REPORT_RESULT_DETAILS[metric]['WARNING'])
                    else:
                        self.results['DANGER'].append(
                            DIAGNOSTIC_REPORT_RESULT_DETAILS[metric]['DANGER'])

    def _get_num_iterations(self, property_name, metadata):
        """Get the number of iterations for the property.

        Args:
            property_name (str):
                The name of the property.
            metadata (dict):
                The metadata of the table.
        """
        self._check_property_name(property_name)

        num_columns = len(metadata['columns'])

        if property_name in ['Coverage', 'Boundary']:
            return num_columns
        elif property_name == 'Synthesis':
            return 1

    def _print_results(self, out=sys.stdout):
        """Print the diagnostic report results."""
        self._generate_results()

        out.write('\nDiagnostic Results:\n')
        print_results_for_level(out, self.results, 'SUCCESS')
        print_results_for_level(out, self.results, 'WARNING')
        print_results_for_level(out, self.results, 'DANGER')

    def generate(self, real_data, synthetic_data, metadata, verbose=True):
        """Generate report.

        Args:
            real_data (pandas.DataFrame):
                The real data.
            synthetic_data (pandas.DataFrame):
                The synthetic data.
            metadata (dict):
                The metadata, which contains each column's data type.
            verbose (bool):
                Whether or not to print report summary and progress.
        """
        super().generate(real_data, synthetic_data, metadata, verbose)
        self._generate_results()

    def get_results(self):
        """Return the diagnostic results.

        Returns:
            dict
                The diagnostic results.
        """
        return deepcopy(self.results)
