import numpy as np
import pandas as pd

from sdmetrics.demos import load_demo
from sdmetrics.reports.multi_table import DiagnosticReport


class TestDiagnosticReport:

    def test_end_to_end(self):
        """Test the end-to-end functionality of the ``DiagnosticReport`` report."""
        real_data, synthetic_data, metadata = load_demo(modality='multi_table')
        report = DiagnosticReport()

        # Run
        report.generate(real_data, synthetic_data, metadata, verbose=False)
        results = report.get_score()

        # Assert
        assert results == 0.9814814814814815

    def test_end_to_end_with_object_datetimes(self):
        """Test the ``DiagnosticReport`` report with object datetimes."""
        real_data, synthetic_data, metadata = load_demo(modality='multi_table')
        for table, table_meta in metadata['tables'].items():
            for column, column_meta in table_meta['columns'].items():
                if column_meta['sdtype'] == 'datetime':
                    dt_format = column_meta['datetime_format']
                    real_data[table][column] = real_data[table][column].dt.strftime(dt_format)

        report = DiagnosticReport()

        # Run
        report.generate(real_data, synthetic_data, metadata, verbose=False)
        results = report.get_score()
        properties = report.get_properties()

        # Assert
        expected_dataframe = pd.DataFrame({
            'Property': ['Data Validity', 'Data Structure', 'Relationship Validity'],
            'Score': [0.9444444444444445, 1.0, 1.0]
        })
        assert results == 0.9814814814814815
        pd.testing.assert_frame_equal(properties, expected_dataframe)

    def test_end_to_end_with_metrics_failing(self):
        """Test the ``DiagnosticReport`` report when some metrics crash.

        This test makes fail the 'Boundary' property to check that the report still works.
        """
        real_data, synthetic_data, metadata = load_demo(modality='multi_table')
        real_data['users']['age'].iloc[0] = 'error_1'
        real_data['transactions']['timestamp'].iloc[0] = 'error_2'
        real_data['transactions']['amount'].iloc[0] = 'error_3'

        report = DiagnosticReport()

        # Run
        report.generate(real_data, synthetic_data, metadata, verbose=False)
        results = report.get_score()

        # Assert
        expected_properties = pd.DataFrame({
            'Property': ['Data Validity', 'Data Structure', 'Relationship Validity'],
            'Score': [0.9677777777777777, 0.7833333333333333, 1.0]
        })
        expected_details = pd.DataFrame({
            'Table': [
                'users', 'users', 'users', 'users', 'sessions', 'sessions', 'sessions',
                'sessions', 'transactions', 'transactions', 'transactions', 'transactions',
                'transactions'
            ],
            'Column': [
                'user_id', 'country', 'gender', 'age', 'session_id', 'user_id', 'device',
                'os', 'transaction_id', 'session_id', 'timestamp', 'amount', 'approved'
            ],
            'Metric': [
                'KeyUniqueness', 'CategoryAdherence', 'CategoryAdherence', 'BoundaryAdherence',
                'KeyUniqueness', 'KeyUniqueness', 'CategoryAdherence', 'CategoryAdherence',
                'KeyUniqueness', 'KeyUniqueness', 'BoundaryAdherence', 'BoundaryAdherence',
                'CategoryAdherence'
            ],
            'Score': [
                1.0, 1.0, 1.0, np.nan, 1.0, 0.7777777777777778, 1.0, 1.0, 1.0, 0.9,
                np.nan, np.nan, 1.0
            ],
            'Error': [
                None, None, None,
                "TypeError: '<=' not supported between instances of 'str' and 'int'",
                None, None, None, None, None, None,
                "TypeError: '<=' not supported between instances of 'str' and 'Timestamp'",
                "TypeError: '<=' not supported between instances of 'str' and 'float'", None
            ]
        })
        assert results == 0.9170370370370371
        pd.testing.assert_frame_equal(
            report.get_properties(), expected_properties, check_exact=False, atol=2e-2
        )
        pd.testing.assert_frame_equal(report.get_details('Data Validity'), expected_details)

    def test_get_properties(self):
        """Test the ``get_properties`` method."""
        # Setup
        real_data, synthetic_data, metadata = load_demo(modality='multi_table')
        report = DiagnosticReport()

        # Run
        report.generate(real_data, synthetic_data, metadata, verbose=False)
        properties = report.get_properties()

        # Assert
        expected_dataframe = pd.DataFrame({
            'Property': ['Data Validity', 'Data Structure', 'Relationship Validity'],
            'Score': [0.9444444444444445, 1.0, 1.0]
        })

        pd.testing.assert_frame_equal(properties, expected_dataframe)

    def test_get_details(self):
        """Test the ``get_details`` method."""
        # Setup
        real_data, synthetic_data, metadata = load_demo(modality='multi_table')
        report = DiagnosticReport()

        # Run
        report.generate(real_data, synthetic_data, metadata, verbose=False)
        details = report.get_details('Data Validity')

        # Assert
        expected_dataframe = pd.DataFrame({
            'Table': [
                'users', 'users', 'users', 'users', 'sessions', 'sessions', 'sessions',
                'sessions', 'transactions', 'transactions', 'transactions', 'transactions',
                'transactions'
            ],
            'Column': [
                'user_id', 'country', 'gender', 'age', 'session_id', 'user_id', 'device',
                'os', 'transaction_id', 'session_id', 'timestamp', 'amount', 'approved'
            ],
            'Metric': [
                'KeyUniqueness', 'CategoryAdherence', 'CategoryAdherence', 'BoundaryAdherence',
                'KeyUniqueness', 'KeyUniqueness', 'CategoryAdherence', 'CategoryAdherence',
                'KeyUniqueness', 'KeyUniqueness', 'BoundaryAdherence', 'BoundaryAdherence',
                'CategoryAdherence'
            ],
            'Score': [
                1.0, 1.0, 1.0, 0.9, 1.0, 0.7777777777777778, 1.0, 1.0, 1.0, 0.9, 0.9,
                0.8, 1.0
            ]
        })

        pd.testing.assert_frame_equal(details, expected_dataframe)

    def test_get_details_with_table_name(self):
        """Test the ``get_details`` method with a table_name parameter."""
        # Setup
        real_data, synthetic_data, metadata = load_demo(modality='multi_table')
        report = DiagnosticReport()

        # Run
        report.generate(real_data, synthetic_data, metadata, verbose=False)
        details = report.get_details('Data Validity', 'users')

        # Assert
        expected_dataframe = pd.DataFrame({
            'Table': ['users', 'users', 'users', 'users'],
            'Column': ['user_id', 'country', 'gender', 'age'],
            'Metric': [
                'KeyUniqueness', 'CategoryAdherence', 'CategoryAdherence', 'BoundaryAdherence'
            ],
            'Score': [1.0, 1.0, 1.0, 0.9]
        })

        pd.testing.assert_frame_equal(details, expected_dataframe)
