import numpy as np
import pandas as pd
import pytest

from sdmetrics.demos import load_demo
from sdmetrics.reports.multi_table import DiagnosticReport
from tests.utils import assert_report_scores_are_not_nan


class TestDiagnosticReport:
    def test_end_to_end(self):
        """Test the end-to-end functionality of the ``DiagnosticReport`` report."""
        real_data, synthetic_data, metadata = load_demo(modality='multi_table')
        report = DiagnosticReport()

        # Run
        report.generate(real_data, synthetic_data, metadata, verbose=False)
        results = report.get_score()

        # Assert
        assert results == 1.0
        assert_report_scores_are_not_nan(report)

    def test_end_to_end_composite_keys(self, composite_keys_multi_table_demo):
        """Test the end-to-end functionality of the ``DiagnosticReport`` report."""
        real_data, synthetic_data, metadata = composite_keys_multi_table_demo
        report = DiagnosticReport()

        # Run
        report.generate(real_data, synthetic_data, metadata, verbose=False)
        results = report.get_score()

        # Assert
        assert results == 1.0
        properties = report.get_properties()
        assert all(properties['Score'] == 1.0)
        assert_report_scores_are_not_nan(report)

    def test_end_to_end_with_metadata_v2(self, metadata_v2_multi_table_demo):
        """Test the diagnostic report with a metadata that defines the range of the columns."""
        # Setup
        real_data, synthetic_data, metadata = metadata_v2_multi_table_demo
        report = DiagnosticReport()

        # Run
        report.generate(real_data, synthetic_data, metadata, verbose=False)
        results = report.get_score()

        # Assert
        assert results == 1.0
        properties = report.get_properties()
        assert all(properties['Score'] == 1.0)
        assert_report_scores_are_not_nan(report)

    def test_end_to_end_metadata_v2_ranges_broader_than_real_data(
        self, metadata_v2_multi_table_demo
    ):
        """Test that the ranges of the metadata are used instead of the ones of the real data."""
        # Setup
        real_data, synthetic_data, metadata_v2 = metadata_v2_multi_table_demo
        metadata_v1 = load_demo(modality='multi_table')[2]

        synthetic_data['users'].loc[0, 'age'] = 80
        synthetic_data['users'].loc[0, 'country'] = 'IT'
        synthetic_data['transactions'].loc[0, 'amount'] = 500.0
        synthetic_data['transactions'].loc[0, 'timestamp'] = pd.Timestamp('2019-06-01')

        report = DiagnosticReport()
        report_v2 = DiagnosticReport()

        # Run
        report.generate(real_data, synthetic_data, metadata_v1, verbose=False)
        report_v2.generate(real_data, synthetic_data, metadata_v2, verbose=False)

        # Assert
        out_of_real_range = [
            ('users', 'age', 'BoundaryAdherence'),
            ('users', 'country', 'CategoryAdherence'),
            ('transactions', 'amount', 'BoundaryAdherence'),
            ('transactions', 'timestamp', 'BoundaryAdherence'),
        ]
        index = ['Table', 'Column', 'Metric']
        scores = report.get_details('Data Validity').set_index(index)['Score']
        scores_v2 = report_v2.get_details('Data Validity').set_index(index)['Score']
        for detail in out_of_real_range:
            assert scores[detail] < 1.0
            assert scores_v2[detail] == 1.0

        assert report.get_score() < 1.0
        assert report_v2.get_score() == 1.0

    def test_end_to_end_with_datetime64_columns(self):
        """Test the ``DiagnosticReport`` report when the datetimes are ``datetime64``."""
        real_data, synthetic_data, metadata = load_demo(modality='multi_table')
        for table, table_meta in metadata['tables'].items():
            for column, column_meta in table_meta['columns'].items():
                if column_meta['sdtype'] == 'datetime':
                    dt_format = column_meta['datetime_format']
                    real_data[table][column] = pd.to_datetime(
                        real_data[table][column], format=dt_format
                    )

        report = DiagnosticReport()

        # Run
        report.generate(real_data, synthetic_data, metadata, verbose=False)
        results = report.get_score()
        properties = report.get_properties()

        # Assert
        expected_dataframe = pd.DataFrame({
            'Property': ['Data Validity', 'Data Structure', 'Relationship Validity'],
            'Score': [1.0, 1.0, 1.0],
        })
        assert results == 1.0
        pd.testing.assert_frame_equal(properties, expected_dataframe)
        assert_report_scores_are_not_nan(report)

    def test_end_to_end_with_metrics_failing(self, converted_datetime_multi_table_demo):
        """Test the ``DiagnosticReport`` report when some metrics crash.

        This test makes fail the 'Boundary' property to check that the report still works.
        The TableStructure should no longer be 1.0 since there is some dtype mismatch.
        """
        real_data, synthetic_data, metadata = converted_datetime_multi_table_demo
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
            'Score': [1.0, 0.6761904761904761, 1.0],
        })
        expected_details = pd.DataFrame({
            'Table': [
                'users',
                'users',
                'users',
                'users',
                'users',
                'sessions',
                'sessions',
                'sessions',
                'sessions',
                'sessions',
                'transactions',
                'transactions',
                'transactions',
                'transactions',
                'transactions',
                'transactions',
                'transactions',
            ],
            'Column': [
                'user_id',
                'user_id',
                'country',
                'gender',
                'age',
                'session_id',
                'session_id',
                'user_id',
                'device',
                'os',
                'transaction_id',
                'transaction_id',
                'session_id',
                'timestamp',
                'timestamp',
                'amount',
                'approved',
            ],
            'Metric': [
                'KeyUniqueness',
                'RegexFormatAdherence',
                'CategoryAdherence',
                'CategoryAdherence',
                'BoundaryAdherence',
                'KeyUniqueness',
                'RegexFormatAdherence',
                'RegexFormatAdherence',
                'CategoryAdherence',
                'CategoryAdherence',
                'KeyUniqueness',
                'RegexFormatAdherence',
                'RegexFormatAdherence',
                'BoundaryAdherence',
                'DatetimeFormatAdherence',
                'BoundaryAdherence',
                'CategoryAdherence',
            ],
            'Score': [
                1.0,
                1.0,
                1.0,
                1.0,
                np.nan,
                1.0,
                1.0,
                1.0,
                1.0,
                1.0,
                1.0,
                1.0,
                1.0,
                np.nan,
                1.0,
                np.nan,
                1.0,
            ],
            'Error': [
                None,
                None,
                None,
                None,
                "TypeError: '<=' not supported between instances of 'str' and 'int'",
                None,
                None,
                None,
                None,
                None,
                None,
                None,
                None,
                "TypeError: '<=' not supported between instances of 'str' and 'Timestamp'",
                None,
                "TypeError: '<=' not supported between instances of 'str' and 'float'",
                None,
            ],
        })
        assert results == 0.892063492063492
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
            'Score': [1.0, 1.0, 1.0],
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
                'users',
                'users',
                'users',
                'users',
                'users',
                'sessions',
                'sessions',
                'sessions',
                'sessions',
                'sessions',
                'transactions',
                'transactions',
                'transactions',
                'transactions',
                'transactions',
                'transactions',
                'transactions',
            ],
            'Column': [
                'user_id',
                'user_id',
                'country',
                'gender',
                'age',
                'session_id',
                'session_id',
                'user_id',
                'device',
                'os',
                'transaction_id',
                'transaction_id',
                'session_id',
                'timestamp',
                'timestamp',
                'amount',
                'approved',
            ],
            'Metric': [
                'KeyUniqueness',
                'RegexFormatAdherence',
                'CategoryAdherence',
                'CategoryAdherence',
                'BoundaryAdherence',
                'KeyUniqueness',
                'RegexFormatAdherence',
                'RegexFormatAdherence',
                'CategoryAdherence',
                'CategoryAdherence',
                'KeyUniqueness',
                'RegexFormatAdherence',
                'RegexFormatAdherence',
                'BoundaryAdherence',
                'DatetimeFormatAdherence',
                'BoundaryAdherence',
                'CategoryAdherence',
            ],
            'Score': [1.0] * 17,
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
            'Table': ['users', 'users', 'users', 'users', 'users'],
            'Column': ['user_id', 'user_id', 'country', 'gender', 'age'],
            'Metric': [
                'KeyUniqueness',
                'RegexFormatAdherence',
                'CategoryAdherence',
                'CategoryAdherence',
                'BoundaryAdherence',
            ],
            'Score': [1.0, 1.0, 1.0, 1.0, 1.0],
        })

        pd.testing.assert_frame_equal(details, expected_dataframe)

    @pytest.mark.filterwarnings('error::UserWarning')
    def test_metadata_without_relationship(self):
        # Setup
        real_data, synthetic_data, metadata = load_demo(modality='multi_table')
        del metadata['relationships']
        report = DiagnosticReport()

        # Run and Assert
        report.generate(real_data, synthetic_data, metadata)
