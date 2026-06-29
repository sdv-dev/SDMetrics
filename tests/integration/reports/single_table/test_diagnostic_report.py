import numpy as np
import pandas as pd
import pytest

from sdmetrics.demos import load_demo
from sdmetrics.reports.single_table import DiagnosticReport


@pytest.fixture
def demo_data_and_metadata():
    real_data, synthetic_data, metadata = load_demo(modality='single_table')
    return real_data['student_placements'], synthetic_data['student_placements'], metadata


class TestDiagnosticReport:
    def test_get_properties(self, demo_data_and_metadata):
        """Test the ``get_properties`` method."""
        # Setup
        real_data, synthetic_data, metadata = demo_data_and_metadata
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

    def test_get_score(self, demo_data_and_metadata):
        """Test the ``get_score`` method."""
        # Setup
        real_data, synthetic_data, metadata = demo_data_and_metadata
        report = DiagnosticReport()

        # Run
        report.generate(real_data, synthetic_data, metadata)
        result = report.get_score()

        # Assert

        assert result == 1.0

    def test_get_score_with_no_verbose(self, demo_data_and_metadata):
        """Test the ``get_score`` method works when verbose=False."""
        # Setup
        real_data, synthetic_data, metadata = demo_data_and_metadata
        report = DiagnosticReport()

        # Run
        report.generate(real_data, synthetic_data, metadata, verbose=False)
        result_dict = report.get_score()

        # Assert
        assert result_dict == 1.0

    def test_end_to_end(self, demo_data_and_metadata):
        """Test the end-to-end functionality of the diagnostic report."""
        # Setup
        real_data, synthetic_data, metadata = demo_data_and_metadata
        report = DiagnosticReport()

        # Run
        report.generate(real_data, synthetic_data, metadata)

        # Assert
        expected_details_data_validity = pd.DataFrame({
            'Column': [
                'start_date',
                'end_date',
                'salary',
                'duration',
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
                'BoundaryAdherence',
                'BoundaryAdherence',
                'BoundaryAdherence',
                'KeyUniqueness',
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
                1.0,
                1.0,
                1.0,
                1.0,
                1.0,
                1.0,
            ],
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

    def test_end_to_end_composite_keys(self, demo_data_and_metadata):
        """Test the end-to-end functionality of the diagnostic report."""
        # Setup
        real_data, synthetic_data, metadata = demo_data_and_metadata

        metadata['tables']['student_placements']['primary_key'] = ['student_id', 'degree_type']
        report = DiagnosticReport()

        # Run
        report.generate(real_data, synthetic_data, metadata)

        # Assert
        expected_details_data_validity = pd.DataFrame({
            'Column': [
                ['student_id', 'degree_type'],
                'start_date',
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
                'BoundaryAdherence',
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
                1.0,
                1.0,
                1.0,
                1.0,
                1.0,
                1.0,
            ],
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

    def test_generate_with_object_datetimes(self, demo_data_and_metadata):
        """Test the diagnostic report with object datetimes."""
        # Setup
        real_data, synthetic_data, metadata = demo_data_and_metadata
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
                'end_date',
                'salary',
                'duration',
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
                'BoundaryAdherence',
                'BoundaryAdherence',
                'BoundaryAdherence',
                'KeyUniqueness',
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
                1.0,
                1.0,
                1.0,
                1.0,
                1.0,
                1.0,
            ],
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

    def test_generate_multiple_times(self, demo_data_and_metadata):
        """The results should be the same both times."""
        # Setup
        real_data, synthetic_data, metadata = demo_data_and_metadata
        report = DiagnosticReport()

        # Run and assert
        report = DiagnosticReport()
        report.generate(real_data, synthetic_data, metadata, verbose=False)

        assert report.get_score() == 1.0
        report.generate(real_data, synthetic_data, metadata)
        assert report.get_score() == 1.0

    def test_get_details_with_errors(self, demo_data_and_metadata):
        """Test the ``get_details`` function of the diagnostic report when there are errors."""
        # Setup
        real_data, synthetic_data, metadata = demo_data_and_metadata
        report = DiagnosticReport()
        real_data['second_perc'] = 'A'

        # Run
        report.generate(real_data, synthetic_data, metadata)

        # Assert
        expected_details = pd.DataFrame({
            'Column': [
                'start_date',
                'end_date',
                'salary',
                'duration',
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
                'BoundaryAdherence',
                'BoundaryAdherence',
                'BoundaryAdherence',
                'KeyUniqueness',
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
