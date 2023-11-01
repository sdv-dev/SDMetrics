"""Single table base report."""
import pickle
import sys
import time
import warnings
from copy import deepcopy
from datetime import datetime
from importlib.metadata import version

import numpy as np
import pandas as pd
import pkg_resources
import tqdm

from sdmetrics.reports.utils import convert_datetime_columns


class BaseReport():
    """Base report class for single table reports.

    This class creates a base report for single-table data.
    """

    def __init__(self):
        self._overall_score = None
        self.is_generated = False
        self._properties = {}
        self._results_handler = None
        self.report_info = {
            'report_type': self.__class__.__name__,
            'generated_date': None,
            'sdmetrics_version': version('sdmetrics')
        }

    def _validate_metadata_matches_data(self, real_data, synthetic_data, metadata):
        """Validate that the metadata matches the data.

        Raise an error if the column metadata does not match the column data.
        """
        real_columns = set(real_data.columns)
        synthetic_columns = set(synthetic_data.columns)
        metadata_columns = set(metadata['columns'].keys())

        missing_data = metadata_columns.difference(real_columns.union(synthetic_columns))
        missing_metadata = real_columns.union(synthetic_columns).difference(metadata_columns)
        missing_columns = missing_data.union(missing_metadata)

        if missing_columns:
            error_message = (
                'The metadata does not match the data. The following columns are missing'
                ' in the real/synthetic data or in the metadata: '
                f"{', '.join(sorted(missing_columns))}"
            )
            raise ValueError(error_message)

    def validate(self, real_data, synthetic_data, metadata):
        """Validate the inputs.

        Args:
            real_data (pandas.DataFrame):
                The real data.
            synthetic_data (pandas.DataFrame):
                The synthetic data.
            metadata (dict):
                The metadata of the table.
        """
        self._validate_metadata_matches_data(real_data, synthetic_data, metadata)

    def _handle_results(self, verbose):
        raise NotImplementedError

    @staticmethod
    def convert_datetimes(real_data, synthetic_data, metadata):
        """Try to convert all datetime columns to datetime dtype.

        Args:
            real_data (pandas.DataFrame):
                The real data.
            synthetic_data (pandas.DataFrame):
                The synthetic data.
            metadata (dict):
                The metadata, which contains each column's data type as well as relationships.
        """
        for column, col_meta in metadata['columns'].items():
            if col_meta['sdtype'] == 'datetime':
                real_col = real_data[column]
                synth_col = synthetic_data[column]
                try:
                    converted_cols = convert_datetime_columns(real_col, synth_col, col_meta)
                    real_data[column], synthetic_data[column] = converted_cols
                except Exception:
                    continue

    def generate(self, real_data, synthetic_data, metadata, verbose=True):
        """Generate report.

        This method generates the report by iterating through each property and calculating
        the score for each property.

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
        if not isinstance(metadata, dict):
            raise TypeError('The provided metadata is not a dictionary.')

        self.validate(real_data, synthetic_data, metadata)
        self.convert_datetimes(real_data, synthetic_data, metadata)

        self.report_info['generated_date'] = datetime.today().strftime('%Y-%m-%d')
        if 'tables' in metadata:
            self.report_info['num_tables'] = len(metadata['tables'])
            self.report_info['num_rows_real_data'] = {
                name: len(table) for name, table in real_data.items()
            }
            self.report_info['num_rows_synthetic_data'] = {
                name: len(table) for name, table in synthetic_data.items()
            }
        else:
            self.report_info['num_rows_real_data'] = len(real_data)
            self.report_info['num_rows_synthetic_data'] = len(synthetic_data)

        scores = []
        progress_bar = None
        if verbose:
            sys.stdout.write('Generating report ...\n')

        start_time = time.time()
        for ind, (property_name, property_instance) in enumerate(self._properties.items()):
            if verbose:
                num_iterations = int(property_instance._get_num_iterations(metadata))
                progress_bar = tqdm.tqdm(total=num_iterations, file=sys.stdout)
                progress_bar.set_description(
                    f'({ind + 1}/{len(self._properties)}) Evaluating {property_name}: '
                )

            score = self._properties[property_name].get_score(
                real_data, synthetic_data, metadata, progress_bar=progress_bar
            )
            scores.append(score)
            if verbose:
                progress_bar.close()

        self._overall_score = np.nanmean(scores)
        self.is_generated = True
        end_time = time.time()
        self.report_info['generation_time'] = end_time - start_time

        self._handle_results(verbose)

    def _check_property_name(self, property_name):
        """Check that the given property name is valid.

        Args:
            property_name (str):
                The name of the property to check.
        """
        if property_name not in self._properties:
            valid_property_names = "', '".join(self._properties.keys())
            raise ValueError(
                f"Invalid property name '{property_name}'."
                f" Valid property names are '{valid_property_names}'."
            )

    def get_info(self):
        """Get the information about the report."""
        return deepcopy(self.report_info)

    def _check_report_generated(self):
        if not self.is_generated:
            raise ValueError('The report has not been generated. Please call `generate` first.')

    def _validate_property_generated(self, property_name):
        """Validate that the given property name and that the report has been generated."""
        self._check_property_name(property_name)
        self._check_report_generated()

    def get_properties(self):
        """Return the property score.

        Returns:
            pandas.DataFrame
                The property score.
        """
        self._check_report_generated()
        name, score = [], []
        for property_name, property_instance in self._properties.items():
            name.append(property_name)
            score.append(property_instance._compute_average())

        return pd.DataFrame({
            'Property': name,
            'Score': score,
        })

    def get_visualization(self, property_name):
        """Return a visualization for each score for the given property name.

        Args:
            property_name (str):
                The name of the property to return score details for.

        Returns:
            plotly.graph_objects._figure.Figure
                The visualization for the requested property.
        """
        self._validate_property_generated(property_name)
        return self._properties[property_name].get_visualization()

    def get_details(self, property_name):
        """Return the details table for the given property name.

        Args:
            property_name (str):
                The name of the property to return details for.

        Returns:
            pandas.DataFrame
        """
        self._validate_property_generated(property_name)
        return self._properties[property_name].details.copy()

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
        """Load a ``Report`` instance from a given path.

        Args:
            filepath (str):
                The path to the file where the report is stored.

        Returns:
            SDMetrics Report:
                The loaded report instance.
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
