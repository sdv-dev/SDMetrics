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

from sdmetrics.reports.single_table.base_report import BaseReport
from sdmetrics.reports.single_table._properties import Coverage, Synthesis, Boundaries
from sdmetrics.reports.utils import (
    DIAGNOSTIC_REPORT_RESULT_DETAILS, aggregate_metric_results, print_results_for_level,
    validate_single_table_inputs)
LOGGER = logging.getLogger(__name__)


class DiagnosticReport(BaseReport):
    """Single table diagnostic report.

    This class creates a diagnostic report for single-table data. It calculates the diagnostic
    score along three properties - synthesis, coverage, and boundaries.
    """

    def __init__(self):
        self._overall_quality_score = None
        self.is_generated = False
        self._properties = {
            'Coverage': Coverage(),
            'Boundaries': Boundaries(),
            'Synthesis': Synthesis()
        }

    def _get_num_iterations(self, property_name, metadata):
        num_columns = len(metadata['columns'])
        if property_name == 'Coverage':
            return num_columns
        elif property_name == 'Boundaries':
            return num_columns
        elif property_name == 'Synthesis':
            return 1

    def _print_results(self, out=sys.stdout):
        """Print the diagnostic report results."""
        self._results['SUCCESS'] = []
        self._results['WARNING'] = []
        self._results['DANGER'] = []

        for property in self._properties:
            details = self._properties[property].get_details()
            average_score_metric = details.groupby('metric').mean()['score']

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
