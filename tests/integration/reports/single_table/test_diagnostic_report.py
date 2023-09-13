import numpy as np
import pandas as pd

from sdmetrics.demos import load_demo
from sdmetrics.reports.single_table import DiagnosticReport


class TestDiagnosticReport:

    def test_get_properties(self):
        """Test the ``get_properties`` method."""
        # Setup
        real_data, synthetic_data, metadata = load_demo(modality='single_table')
        report = DiagnosticReport()

        # Run
        report.generate(real_data, synthetic_data, metadata)
        properties_frame = report.get_properties()

        # Assert
        expected_frame = pd.DataFrame(
            {
                'Property': ['Coverage', 'Boundary', 'Synthesis'],
                'Score': [0.9419212095491987, 0.9172655676537751, 1.0]
            }
        )
        pd.testing.assert_frame_equal(properties_frame, expected_frame)

    def test_get_results(self):
        """Test the ``get_results`` method."""
        # Setup
        real_data, synthetic_data, metadata = load_demo(modality='single_table')
        report = DiagnosticReport()

        # Run
        report.generate(real_data, synthetic_data, metadata)
        result_dict = report.get_results()

        # Assert
        expected_results = {
            'SUCCESS': [
                'The synthetic data covers over 90% of the categories present in the real data',
                'The synthetic data covers over 90% of the numerical ranges present in the '
                'real data',
                'The synthetic data follows over 90% of the min/max boundaries set by the '
                'real data',
                'Over 90% of the synthetic rows are not copies of the real data'
            ],
            'WARNING': [],
            'DANGER': []
        }

        assert result_dict == expected_results

    def test_get_results_with_no_verbose(self):
        """Test the ``get_results`` method works when verbose=False."""
        # Setup
        real_data, synthetic_data, metadata = load_demo(modality='single_table')
        report = DiagnosticReport()

        # Run
        report.generate(real_data, synthetic_data, metadata, verbose=False)
        result_dict = report.get_results()

        # Assert
        expected_results = {
            'SUCCESS': [
                'The synthetic data covers over 90% of the categories present in the real data',
                'The synthetic data covers over 90% of the numerical ranges present in the '
                'real data',
                'The synthetic data follows over 90% of the min/max boundaries set by the '
                'real data',
                'Over 90% of the synthetic rows are not copies of the real data'
            ],
            'WARNING': [],
            'DANGER': []
        }

        assert result_dict == expected_results

    def test_end_to_end(self):
        """Test the end-to-end functionality of the diagnostic report."""
        # Setup
        real_data, synthetic_data, metadata = load_demo(modality='single_table')
        report = DiagnosticReport()

        # Run
        report.generate(real_data, synthetic_data, metadata)

        # Assert
        expected_details_synthetis = pd.DataFrame(
            {
                'Metric': 'NewRowSynthesis',
                'Score': 1.0,
                'Num Matched Rows': 0,
                'Num New Rows': 215
            }, index=[0]
        )

        expected_details_coverage = pd.DataFrame({
            'Column': [
                'start_date', 'end_date', 'salary', 'duration', 'high_perc', 'high_spec',
                'mba_spec', 'second_perc', 'gender', 'degree_perc', 'placed', 'experience_years',
                'employability_perc', 'mba_perc', 'work_experience', 'degree_type'
            ],
            'Metric': [
                'RangeCoverage', 'RangeCoverage', 'RangeCoverage', 'RangeCoverage',
                'RangeCoverage', 'CategoryCoverage', 'CategoryCoverage', 'RangeCoverage',
                'CategoryCoverage', 'RangeCoverage', 'CategoryCoverage', 'RangeCoverage',
                'RangeCoverage', 'RangeCoverage', 'CategoryCoverage', 'CategoryCoverage'
            ],
            'Score': [
                1.0, 1.0, 0.42333783783783785, 1.0, 0.9807348482826732, 1.0, 1.0, 1.0, 1.0, 1.0,
                1.0, 0.6666666666666667, 1.0, 1.0, 1.0, 1.0
            ]
        })

        expected_details_boundary = pd.DataFrame({
            'Column': [
                'start_date', 'end_date', 'salary', 'duration', 'high_perc', 'second_perc',
                'degree_perc', 'experience_years', 'employability_perc', 'mba_perc'
            ],
            'Metric': ['BoundaryAdherence'] * 10,
            'Score': [
                0.8503937007874016, 0.8615384615384616, 0.9444444444444444, 1.0,
                0.8651162790697674, 0.9255813953488372, 0.9441860465116279, 1.0,
                0.8883720930232558, 0.8930232558139535
            ]
        })

        pd.testing.assert_frame_equal(
            report.get_details('Synthesis'),
            expected_details_synthetis
        )

        pd.testing.assert_frame_equal(
            report.get_details('Coverage'),
            expected_details_coverage
        )

        pd.testing.assert_frame_equal(
            report.get_details('Boundary'),
            expected_details_boundary
        )

    def test_generate_with_object_datetimes(self):
        """Test the diagnostic report with object datetimes."""
        # Setup
        real_data, synthetic_data, metadata = load_demo(modality='single_table')
        for column, column_meta in metadata['columns'].items():
            if column_meta['sdtype'] == 'datetime':
                dt_format = column_meta['datetime_format']
                real_data[column] = real_data[column].dt.strftime(dt_format)

        report = DiagnosticReport()

        # Run
        report.generate(real_data, synthetic_data, metadata)

        # Assert
        expected_details_synthetis = pd.DataFrame(
            {
                'Metric': 'NewRowSynthesis',
                'Score': 1.0,
                'Num Matched Rows': 0,
                'Num New Rows': 215
            }, index=[0]
        )

        expected_details_coverage = pd.DataFrame({
            'Column': [
                'start_date', 'end_date', 'salary', 'duration', 'high_perc', 'high_spec',
                'mba_spec', 'second_perc', 'gender', 'degree_perc', 'placed', 'experience_years',
                'employability_perc', 'mba_perc', 'work_experience', 'degree_type'
            ],
            'Metric': [
                'RangeCoverage', 'RangeCoverage', 'RangeCoverage', 'RangeCoverage',
                'RangeCoverage', 'CategoryCoverage', 'CategoryCoverage', 'RangeCoverage',
                'CategoryCoverage', 'RangeCoverage', 'CategoryCoverage', 'RangeCoverage',
                'RangeCoverage', 'RangeCoverage', 'CategoryCoverage', 'CategoryCoverage'
            ],
            'Score': [
                1.0, 1.0, 0.42333783783783785, 1.0, 0.9807348482826732, 1.0, 1.0, 1.0, 1.0, 1.0,
                1.0, 0.6666666666666667, 1.0, 1.0, 1.0, 1.0
            ]
        })

        expected_details_boundary = pd.DataFrame({
            'Column': [
                'start_date', 'end_date', 'salary', 'duration', 'high_perc', 'second_perc',
                'degree_perc', 'experience_years', 'employability_perc', 'mba_perc'
            ],
            'Metric': ['BoundaryAdherence'] * 10,
            'Score': [
                0.8503937007874016, 0.8615384615384616, 0.9444444444444444, 1.0,
                0.8651162790697674, 0.9255813953488372, 0.9441860465116279, 1.0,
                0.8883720930232558, 0.8930232558139535
            ]
        })

        pd.testing.assert_frame_equal(
            report.get_details('Synthesis'),
            expected_details_synthetis
        )

        pd.testing.assert_frame_equal(
            report.get_details('Coverage'),
            expected_details_coverage
        )

        pd.testing.assert_frame_equal(
            report.get_details('Boundary'),
            expected_details_boundary
        )

    def test_generate_multiple_times(self):
        """The results should be the same both times."""
        # Setup
        real_data, synthetic_data, metadata = load_demo(modality='single_table')
        report = DiagnosticReport()

        # Run and assert
        report = DiagnosticReport()
        report.generate(real_data, synthetic_data, metadata, verbose=False)
        expected_results = {
            'DANGER': [],
            'SUCCESS': [
                'The synthetic data covers over 90% of the categories present in the real data',
                'The synthetic data covers over 90% of the numerical ranges present in the real '
                'data',
                'The synthetic data follows over 90% of the min/max boundaries set by the real '
                'data',
                'Over 90% of the synthetic rows are not copies of the real data'
            ],
            'WARNING': []
        }
        assert report.get_results() == expected_results
        report.generate(real_data, synthetic_data, metadata)
        assert report.get_results() == expected_results

    def test_get_details_with_errors(self):
        """Test the ``get_details`` function of the diagnostic report when there are errors."""
        # Setup
        real_data, synthetic_data, metadata = load_demo(modality='single_table')
        report = DiagnosticReport()
        real_data['second_perc'] = np.nan

        # Run
        report.generate(real_data, synthetic_data, metadata)

        # Assert
        expected_details = pd.DataFrame({
            'Column': {
                0: 'start_date',
                1: 'end_date',
                2: 'salary',
                3: 'duration',
                4: 'high_perc',
                5: 'second_perc',
                6: 'degree_perc',
                7: 'experience_years',
                8: 'employability_perc',
                9: 'mba_perc'
            },
            'Metric': {
                0: 'BoundaryAdherence',
                1: 'BoundaryAdherence',
                2: 'BoundaryAdherence',
                3: 'BoundaryAdherence',
                4: 'BoundaryAdherence',
                5: 'BoundaryAdherence',
                6: 'BoundaryAdherence',
                7: 'BoundaryAdherence',
                8: 'BoundaryAdherence',
                9: 'BoundaryAdherence'
            },
            'Score': {
                0: 0.8503937007874016,
                1: 0.8615384615384616,
                2: 0.9444444444444444,
                3: 1.0,
                4: 0.8651162790697674,
                5: np.nan,
                6: 0.9441860465116279,
                7: 1.0,
                8: 0.8883720930232558,
                9: 0.8930232558139535
            },
            'Error': {
                0: None,
                1: None,
                2: None,
                3: None,
                4: None,
                5: 'InvalidDataError: All NaN values in real data.',
                6: None,
                7: None,
                8: None,
                9: None
            }
        })
        pd.testing.assert_frame_equal(
            report.get_details('Boundary'),
            expected_details
        )
