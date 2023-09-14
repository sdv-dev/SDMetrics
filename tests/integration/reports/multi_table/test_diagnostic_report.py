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
        results = report.get_results()

        # Assert
        expected_results = {
            'SUCCESS': [
                'The synthetic data covers over 90% of the categories present in the real data',
                'The synthetic data covers over 90% of the numerical ranges present'
                ' in the real data'
            ],
            'WARNING': [
                'More than 10% the synthetic data does not follow the min/max boundaries'
                ' set by the real data',
                'More than 10% of the synthetic rows are copies of the real data'
            ],
            'DANGER': []
        }
        assert results == expected_results

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
        results = report.get_results()
        properties = report.get_properties()

        # Assert
        expected_dataframe = pd.DataFrame({
            'Property': ['Coverage', 'Boundary', 'Synthesis'],
            'Score': [0.9573447196980541, 0.8666666666666667, 0.6333333333333333]
        })
        expected_results = {
            'SUCCESS': [
                'The synthetic data covers over 90% of the categories present in the real data',
                'The synthetic data covers over 90% of the numerical ranges present'
                ' in the real data'
            ],
            'WARNING': [
                'More than 10% the synthetic data does not follow the min/max boundaries'
                ' set by the real data',
                'More than 10% of the synthetic rows are copies of the real data'
            ],
            'DANGER': []
        }
        assert results == expected_results
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
        results = report.get_results()

        # Assert
        expected_results = {
            'SUCCESS': [
                'The synthetic data covers over 90% of the categories present in the real data'
            ],
            'WARNING': [],
            'DANGER': ['More than 50% of the synthetic rows are copies of the real data']
        }
        expected_properties = pd.DataFrame({
            'Property': ['Coverage', 'Boundary', 'Synthesis'],
            'Score': [0.9666666666666668, np.nan, 0.0]
        })
        expected_details = pd.DataFrame({
            'Table': ['users', 'transactions', 'transactions'],
            'Column': ['age', 'timestamp', 'amount'],
            'Metric': ['BoundaryAdherence', 'BoundaryAdherence', 'BoundaryAdherence'],
            'Score': [np.nan, np.nan, np.nan],
            'Error': [
                "TypeError: '<=' not supported between instances of 'str' and 'int'",
                "TypeError: '<=' not supported between instances of 'str' and 'Timestamp'",
                "TypeError: '<=' not supported between instances of 'str' and 'float'"
            ]
        })
        assert results == expected_results
        pd.testing.assert_frame_equal(
            report.get_properties(), expected_properties, check_exact=False, atol=2e-2
        )
        pd.testing.assert_frame_equal(report.get_details('Boundary'), expected_details)

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
            'Property': ['Coverage', 'Boundary', 'Synthesis'],
            'Score': [0.9573447196980541, 0.8666666666666667, 0.6333333333333333]
        })

        pd.testing.assert_frame_equal(properties, expected_dataframe)

    def test_get_details(self):
        """Test the ``get_properties`` method."""
        # Setup
        real_data, synthetic_data, metadata = load_demo(modality='multi_table')
        report = DiagnosticReport()

        # Run
        report.generate(real_data, synthetic_data, metadata, verbose=False)
        details = report.get_details('Coverage')

        # Assert
        expected_dataframe = pd.DataFrame({
            'Table': [
                'users', 'users', 'users', 'sessions', 'sessions', 'transactions',
                'transactions', 'transactions'
            ],
            'Column': [
                'country', 'gender', 'age', 'device', 'os', 'timestamp',
                'amount', 'approved'
            ],
            'Metric': [
                'CategoryCoverage', 'CategoryCoverage', 'RangeCoverage', 'CategoryCoverage',
                'CategoryCoverage', 'RangeCoverage', 'RangeCoverage', 'CategoryCoverage'
            ],
            'Score': [
                0.8333333333333334, 1.0, 1.0, 1.0, 1.0, 0.9955955390408375,
                0.829828885210262, 1.0
            ]
        })

        pd.testing.assert_frame_equal(details, expected_dataframe)

    def test_get_details_with_table_name(self):
        """Test the ``get_properties`` method with a table_name parameter."""
        # Setup
        real_data, synthetic_data, metadata = load_demo(modality='multi_table')
        report = DiagnosticReport()

        # Run
        report.generate(real_data, synthetic_data, metadata, verbose=False)
        details = report.get_details('Coverage', 'users')

        # Assert
        expected_dataframe = pd.DataFrame({
            'Table': ['users', 'users', 'users'],
            'Column': ['country', 'gender', 'age'],
            'Metric': ['CategoryCoverage', 'CategoryCoverage', 'RangeCoverage'],
            'Score': [0.8333333333333334, 1.0, 1.0]
        })

        pd.testing.assert_frame_equal(details, expected_dataframe)
