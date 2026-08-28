import numpy as np
import pandas as pd

from sdmetrics.reports.single_table import DiagnosticReport
from tests.utils import assert_report_scores_are_not_nan


class TestDiagnosticReport:
    def test_get_properties(self, single_table_demo_data_and_metadata):
        """Test the ``get_properties`` method."""
        # Setup
        real_data, synthetic_data, metadata = single_table_demo_data_and_metadata
        report = DiagnosticReport()

        # Run
        report.generate(real_data, synthetic_data, metadata)
        properties_frame = report.get_properties()

        # Assert
        expected_frame = pd.DataFrame({
            'Property': ['Data Validity', 'Data Structure'],
            'Score': [1.0, 1.0],
        })
        pd.testing.assert_frame_equal(properties_frame, expected_frame)

    def test_get_score(self, single_table_demo_data_and_metadata):
        """Test the ``get_score`` method."""
        # Setup
        real_data, synthetic_data, metadata = single_table_demo_data_and_metadata
        report = DiagnosticReport()

        # Run
        report.generate(real_data, synthetic_data, metadata)
        result = report.get_score()

        # Assert

        assert result == 1.0

    def test_get_score_with_no_verbose(self, single_table_demo_data_and_metadata):
        """Test the ``get_score`` method works when verbose=False."""
        # Setup
        real_data, synthetic_data, metadata = single_table_demo_data_and_metadata
        report = DiagnosticReport()

        # Run
        report.generate(real_data, synthetic_data, metadata, verbose=False)
        result_dict = report.get_score()

        # Assert
        assert result_dict == 1.0

    def test_end_to_end(self, single_table_demo_data_and_metadata):
        """Test the end-to-end functionality of the diagnostic report."""
        # Setup
        real_data, synthetic_data, metadata = single_table_demo_data_and_metadata
        report = DiagnosticReport()

        # Run
        report.generate(real_data, synthetic_data, metadata)

        # Assert
        expected_details_data_validity = pd.DataFrame({
            'Column': [
                'start_date',
                'start_date',
                'end_date',
                'end_date',
                'salary',
                'duration',
                'student_id',
                'student_id',
                'high_perc',
                'high_spec',
                'mba_spec',
                'second_perc',
                'gender',
                'degree_perc',
                'placed',
                'experience_years',
                'employability_perc',
                'mba_perc',
                'work_experience',
                'degree_type',
            ],
            'Metric': [
                'BoundaryAdherence',
                'DatetimeFormatAdherence',
                'BoundaryAdherence',
                'DatetimeFormatAdherence',
                'BoundaryAdherence',
                'BoundaryAdherence',
                'KeyUniqueness',
                'RegexFormatAdherence',
                'BoundaryAdherence',
                'CategoryAdherence',
                'CategoryAdherence',
                'BoundaryAdherence',
                'CategoryAdherence',
                'BoundaryAdherence',
                'CategoryAdherence',
                'BoundaryAdherence',
                'BoundaryAdherence',
                'BoundaryAdherence',
                'CategoryAdherence',
                'CategoryAdherence',
            ],
            'Score': [1.0] * 20,
        })
        expected_details_data_structure = pd.DataFrame({
            'Metric': ['TableStructure'],
            'Score': [1.0],
        })

        pd.testing.assert_frame_equal(
            report.get_details('Data Validity'), expected_details_data_validity
        )

        pd.testing.assert_frame_equal(
            report.get_details('Data Structure'), expected_details_data_structure
        )
        assert_report_scores_are_not_nan(report)

    def test_end_to_end_with_metadata_v2(self, metadata_v2_single_table_demo):
        """Test the diagnostic report with a metadata that defines the range of the columns."""
        # Setup
        real_data, synthetic_data, metadata = metadata_v2_single_table_demo
        report = DiagnosticReport()

        # Run
        report.generate(real_data, synthetic_data, metadata, verbose=False)

        # Assert
        details = report.get_details('Data Validity')
        assert set(details['Metric']) == {
            'BoundaryAdherence',
            'CategoryAdherence',
            'DatetimeFormatAdherence',
            'KeyUniqueness',
            'RegexFormatAdherence',
        }
        assert (details['Score'] == 1.0).all()
        assert report.get_score() == 1.0
        assert_report_scores_are_not_nan(report)

    def test_end_to_end_metadata_v2_ranges_broader_than_real_data(
        self, single_table_demo_data_and_metadata, metadata_v2_single_table_demo
    ):
        """Test that the ranges of the metadata are used instead of the ones of the real data.

        Synthetic values that are outside the range of the real data but inside the range
        defined by the metadata should be considered valid.
        """
        # Setup
        real_data, synthetic_data, metadata = single_table_demo_data_and_metadata
        real_data_v2, synthetic_data_v2, metadata_v2 = metadata_v2_single_table_demo
        for synthetic_table in [synthetic_data, synthetic_data_v2]:
            synthetic_table.loc[0, 'salary'] = 150000.0
            synthetic_table.loc[0, 'second_perc'] = 99.5
            synthetic_table.loc[0, 'start_date'] = pd.Timestamp('2021-06-01')
            synthetic_table.loc[0, 'high_spec'] = 'Engineering'

        report = DiagnosticReport()
        report_v2 = DiagnosticReport()

        # Run
        report.generate(real_data, synthetic_data, metadata, verbose=False)
        report_v2.generate(real_data_v2, synthetic_data_v2, metadata_v2, verbose=False)

        # Assert
        out_of_real_range = [
            ('salary', 'BoundaryAdherence'),
            ('second_perc', 'BoundaryAdherence'),
            ('start_date', 'BoundaryAdherence'),
            ('high_spec', 'CategoryAdherence'),
        ]
        scores = report.get_details('Data Validity').set_index(['Column', 'Metric'])['Score']
        scores_v2 = report_v2.get_details('Data Validity').set_index(['Column', 'Metric'])['Score']
        for column_name, metric_name in out_of_real_range:
            assert scores[(column_name, metric_name)] < 1.0
            assert scores_v2[(column_name, metric_name)] == 1.0

        assert report.get_score() < 1.0
        assert report_v2.get_score() == 1.0

    def test_end_to_end_composite_keys(self, single_table_demo_data_and_metadata):
        """Test the end-to-end functionality of the diagnostic report."""
        # Setup
        real_data, synthetic_data, metadata = single_table_demo_data_and_metadata

        metadata['tables']['student_placements']['primary_key'] = ['student_id', 'degree_type']
        report = DiagnosticReport()

        # Run
        report.generate(real_data, synthetic_data, metadata)

        # Assert
        expected_details_data_validity = pd.DataFrame({
            'Column': [
                ['student_id', 'degree_type'],
                'start_date',
                'start_date',
                'end_date',
                'end_date',
                'salary',
                'duration',
                'high_perc',
                'high_spec',
                'mba_spec',
                'second_perc',
                'gender',
                'degree_perc',
                'placed',
                'experience_years',
                'employability_perc',
                'mba_perc',
                'work_experience',
                'degree_type',
            ],
            'Metric': [
                'KeyUniqueness',
                'BoundaryAdherence',
                'DatetimeFormatAdherence',
                'BoundaryAdherence',
                'DatetimeFormatAdherence',
                'BoundaryAdherence',
                'BoundaryAdherence',
                'BoundaryAdherence',
                'CategoryAdherence',
                'CategoryAdherence',
                'BoundaryAdherence',
                'CategoryAdherence',
                'BoundaryAdherence',
                'CategoryAdherence',
                'BoundaryAdherence',
                'BoundaryAdherence',
                'BoundaryAdherence',
                'CategoryAdherence',
                'CategoryAdherence',
            ],
            'Score': [1.0] * 19,
        })
        expected_details_data_structure = pd.DataFrame({
            'Metric': ['TableStructure'],
            'Score': [1.0],
        })

        pd.testing.assert_frame_equal(
            report.get_details('Data Validity'), expected_details_data_validity
        )

        pd.testing.assert_frame_equal(
            report.get_details('Data Structure'), expected_details_data_structure
        )
        assert_report_scores_are_not_nan(report)

    def test_generate_with_object_datetimes(self, single_table_demo_data_and_metadata):
        """Test the diagnostic report with object datetimes."""
        # Setup
        real_data, synthetic_data, metadata = single_table_demo_data_and_metadata
        for column, column_meta in metadata['tables']['student_placements']['columns'].items():
            if column_meta['sdtype'] == 'datetime':
                dt_format = column_meta['datetime_format']
                real_data[column] = real_data[column].dt.strftime(dt_format)

        report = DiagnosticReport()

        # Run
        report.generate(real_data, synthetic_data, metadata)

        # Assert
        expected_details_data_validity = pd.DataFrame({
            'Column': [
                'start_date',
                'start_date',
                'end_date',
                'end_date',
                'salary',
                'duration',
                'student_id',
                'student_id',
                'high_perc',
                'high_spec',
                'mba_spec',
                'second_perc',
                'gender',
                'degree_perc',
                'placed',
                'experience_years',
                'employability_perc',
                'mba_perc',
                'work_experience',
                'degree_type',
            ],
            'Metric': [
                'BoundaryAdherence',
                'DatetimeFormatAdherence',
                'BoundaryAdherence',
                'DatetimeFormatAdherence',
                'BoundaryAdherence',
                'BoundaryAdherence',
                'KeyUniqueness',
                'RegexFormatAdherence',
                'BoundaryAdherence',
                'CategoryAdherence',
                'CategoryAdherence',
                'BoundaryAdherence',
                'CategoryAdherence',
                'BoundaryAdherence',
                'CategoryAdherence',
                'BoundaryAdherence',
                'BoundaryAdherence',
                'BoundaryAdherence',
                'CategoryAdherence',
                'CategoryAdherence',
            ],
            'Score': [1.0] * 20,
        })

        expected_details_data_structure = pd.DataFrame({
            'Metric': ['TableStructure'],
            'Score': [1.0],
        })

        pd.testing.assert_frame_equal(
            report.get_details('Data Validity'), expected_details_data_validity
        )

        pd.testing.assert_frame_equal(
            report.get_details('Data Structure'), expected_details_data_structure
        )

    def test_generate_multiple_times(self, single_table_demo_data_and_metadata):
        """The results should be the same both times."""
        # Setup
        real_data, synthetic_data, metadata = single_table_demo_data_and_metadata
        report = DiagnosticReport()

        # Run and assert
        report = DiagnosticReport()
        report.generate(real_data, synthetic_data, metadata, verbose=False)

        assert report.get_score() == 1.0
        report.generate(real_data, synthetic_data, metadata)
        assert report.get_score() == 1.0

    def test_get_details_with_errors(self, single_table_demo_data_and_metadata):
        """Test the ``get_details`` function of the diagnostic report when there are errors."""
        # Setup
        real_data, synthetic_data, metadata = single_table_demo_data_and_metadata
        report = DiagnosticReport()
        real_data['second_perc'] = 'A'

        # Run
        report.generate(real_data, synthetic_data, metadata)

        # Assert
        expected_details = pd.DataFrame({
            'Column': [
                'start_date',
                'start_date',
                'end_date',
                'end_date',
                'salary',
                'duration',
                'student_id',
                'student_id',
                'high_perc',
                'high_spec',
                'mba_spec',
                'second_perc',
                'gender',
                'degree_perc',
                'placed',
                'experience_years',
                'employability_perc',
                'mba_perc',
                'work_experience',
                'degree_type',
            ],
            'Metric': [
                'BoundaryAdherence',
                'DatetimeFormatAdherence',
                'BoundaryAdherence',
                'DatetimeFormatAdherence',
                'BoundaryAdherence',
                'BoundaryAdherence',
                'KeyUniqueness',
                'RegexFormatAdherence',
                'BoundaryAdherence',
                'CategoryAdherence',
                'CategoryAdherence',
                'BoundaryAdherence',
                'CategoryAdherence',
                'BoundaryAdherence',
                'CategoryAdherence',
                'BoundaryAdherence',
                'BoundaryAdherence',
                'BoundaryAdherence',
                'CategoryAdherence',
                'CategoryAdherence',
            ],
            'Score': [
                1.0,
                1.0,
                1.0,
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
                1.0,
                1.0,
                1.0,
                1.0,
                1.0,
                1.0,
                1.0,
            ],
            'Error': [
                None,
                None,
                None,
                None,
                None,
                None,
                None,
                None,
                None,
                None,
                None,
                'TypeError: Invalid comparison between dtype=float64 and str',
                None,
                None,
                None,
                None,
                None,
                None,
                None,
                None,
            ],
        })
        pd.testing.assert_frame_equal(report.get_details('Data Validity'), expected_details)

    def test_report_runs_with_mismatch_data_metadata(self):
        """Test that the report runs with mismatched data and metadata."""
        # Setup
        data = pd.DataFrame({'id': [0, 1, 2], 'val1': ['a', 'a', 'b'], 'val2': [0.1, 2.4, 5.7]})
        synthetic_data = pd.DataFrame({
            'id': [1, 2, 3],
            'extra_col': ['x', 'y', 'z'],
            'val1': ['c', 'd', 'd'],
        })

        metadata = {
            'tables': {
                'table': {
                    'columns': {
                        'id': {'sdtype': 'id'},
                        'val1': {'sdtype': 'categorical'},
                        'val2': {'sdtype': 'numerical'},
                    },
                    'primary_key': 'id',
                },
            },
        }
        report = DiagnosticReport()

        # Run
        report.generate(data, synthetic_data, metadata)

        # Assert
        expected_properties = pd.DataFrame({
            'Property': ['Data Validity', 'Data Structure'],
            'Score': [0.5, 0.5],
        })
        assert report.get_score() == 0.5
        pd.testing.assert_frame_equal(report.get_properties(), expected_properties)
