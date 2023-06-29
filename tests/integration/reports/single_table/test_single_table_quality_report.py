from datetime import date, datetime

import numpy as np
import pandas as pd

from sdmetrics.demos import load_demo
from sdmetrics.reports.single_table import QualityReport


class TestQualityReport:

    def test__get_properties(self):
        """Test the properties of the report."""
        # Setup
        real_data = pd.DataFrame({
            'col1': [0, 1, 2, 3],
            'col2': ['a', 'b', 'c', 'd'],
            'col3': [True, False, False, True],
            'col4': [
                datetime(2020, 10, 1),
                datetime(2021, 1, 2),
                datetime(2021, 9, 12),
                datetime(2022, 10, 1),
            ],
            'col5': [date(2020, 9, 13), date(2020, 12, 1), date(2021, 1, 12), date(2022, 8, 13)],
        })

        synthetic_data = pd.DataFrame({
            'col1': [0, 2, 2, 3],
            'col2': ['a', 'c', 'c', 'b'],
            'col3': [False, False, False, True],
            'col4': [
                datetime(2020, 11, 4),
                datetime(2021, 2, 1),
                datetime(2021, 8, 1),
                datetime(2022, 12, 1),
            ],
            'col5': [date(2020, 10, 13), date(2020, 2, 4), date(2021, 3, 11), date(2022, 7, 23)],
        })

        metadata = {
            'columns': {
                'col1': {'sdtype': 'numerical'},
                'col2': {'sdtype': 'categorical'},
                'col3': {'sdtype': 'boolean'},
                'col4': {'sdtype': 'datetime', 'datetime_format': '%Y-%m-%d'},
                'col5': {'sdtype': 'datetime', 'datetime_format': '%Y-%m-%d'},
            }
        }

        report = QualityReport()

        # Run
        report.generate(real_data, synthetic_data, metadata)
        properties = report.get_properties()

        # Assert
        pd.testing.assert_frame_equal(properties, pd.DataFrame({
            'Property': ['Column Shapes', 'Column Pair Trends'],
            'Score': [0.750000, 0.5],
        }))

    def test_report_end_to_end(self):
        """Test the quality report end to end.

        The report must compute each property and the overall quality score.
        """
        # Setup
        column_names = [
            'student_id', 'degree_type', 'start_date', 'second_perc', 'work_experience'
        ]
        real_data, synthetic_data, metadata = load_demo(modality='single_table')

        metadata['columns'] = {
            key: val for key, val in metadata['columns'].items() if key in column_names
        }
        report = QualityReport()

        # Run
        report.generate(real_data[column_names], synthetic_data[column_names], metadata)

        # Assert
        expected_details_column_shapes_dict = {
            'Column': ['start_date', 'second_perc', 'work_experience', 'degree_type'],
            'Metric': ['KSComplement', 'KSComplement', 'TVComplement', 'TVComplement'],
            'Score': [
                0.7011066184294531, 0.627906976744186, 0.9720930232558139, 0.9255813953488372
            ],
        }

        expected_details_cpt__dict = {
            'Column 1': [
                'start_date', 'start_date', 'start_date', 'second_perc',
                'second_perc', 'work_experience'
            ],
            'Column 2': [
                'second_perc', 'work_experience', 'degree_type', 'work_experience',
                'degree_type', 'degree_type'
            ],
            'Metric': [
                'CorrelationSimilarity', 'ContingencySimilarity', 'ContingencySimilarity',
                'ContingencySimilarity', 'ContingencySimilarity', 'ContingencySimilarity'
            ],
            'Score': [
                0.9854510263003199, 0.586046511627907, 0.6232558139534884, 0.7348837209302326,
                0.6976744186046512, 0.8976744186046511
            ],
            'Real Correlation': [
                0.04735340044317632, np.nan, np.nan, np.nan, np.nan, np.nan
            ],
            'Synthetic Correlation': [
                0.07645134784253645, np.nan, np.nan, np.nan, np.nan, np.nan
            ]
        }
        expected_details_column_shapes = pd.DataFrame(expected_details_column_shapes_dict)
        expected_details_cpt = pd.DataFrame(expected_details_cpt__dict)

        pd.testing.assert_frame_equal(
            report.get_details('Column Shapes'), expected_details_column_shapes
        )
        pd.testing.assert_frame_equal(
            report.get_details('Column Pair Trends'), expected_details_cpt
        )
        assert report.get_score() == 0.78

    def test_report_end_to_end_with_errors(self):
        """Test the quality report end to end with errors in the proerties computation."""
        # Setup
        column_names = [
            'student_id', 'degree_type', 'start_date', 'second_perc', 'work_experience'
        ]
        real_data, synthetic_data, metadata = load_demo(modality='single_table')

        metadata['columns'] = {
            key: val for key, val in metadata['columns'].items() if key in column_names
        }

        real_data['second_perc'].iloc[2] = 'a'

        report = QualityReport()

        # Run
        report.generate(real_data[column_names], synthetic_data[column_names], metadata)

        # Assert
        expected_details_column_shapes_dict = {
            'Column': ['start_date', 'second_perc', 'work_experience', 'degree_type'],
            'Metric': ['KSComplement', 'KSComplement', 'TVComplement', 'TVComplement'],
            'Score': [0.7011066184294531, np.nan, 0.9720930232558139, 0.9255813953488372],
            'Error': [
                None,
                "Error: TypeError '<' not supported between instances of 'str' and 'float'",
                None,
                None
            ]
        }

        expected_details_cpt__dict = {
            'Column 1': [
                'start_date', 'start_date', 'start_date', 'second_perc',
                'second_perc', 'work_experience'
            ],
            'Column 2': [
                'second_perc', 'work_experience', 'degree_type', 'work_experience',
                'degree_type', 'degree_type'
            ],
            'Metric': [
                'CorrelationSimilarity', 'ContingencySimilarity', 'ContingencySimilarity',
                'ContingencySimilarity', 'ContingencySimilarity', 'ContingencySimilarity'
            ],
            'Score': [
                np.nan, 0.586046511627907, 0.6232558139534884, np.nan, np.nan, 0.8976744186046511
            ],
            'Real Correlation': [np.nan] * 6,
            'Synthetic Correlation': [np.nan] * 6,
            'Error': [
                "Error: ValueError could not convert string to float: 'a'",
                None,
                None,
                "Error: TypeError '<=' not supported between instances of 'float' and 'str'",
                "Error: TypeError '<=' not supported between instances of 'float' and 'str'",
                None
            ]
        }
        expected_details_column_shapes = pd.DataFrame(expected_details_column_shapes_dict)
        expected_details_cpt = pd.DataFrame(expected_details_cpt__dict)

        pd.testing.assert_frame_equal(
            report.get_details('Column Shapes'), expected_details_column_shapes
        )
        pd.testing.assert_frame_equal(
            report.get_details('Column Pair Trends'), expected_details_cpt
        )
        assert report.get_score() == 0.7849999999999999

    def test_report_with_column_nan(self):
        """Test the report with column full of NaNs."""
        # Setup
        column_names = [
            'student_id', 'degree_type', 'start_date', 'second_perc', 'work_experience'
        ]
        real_data, synthetic_data, metadata = load_demo(modality='single_table')

        metadata['columns'] = {
            key: val for key, val in metadata['columns'].items() if key in column_names
        }

        real_data['nan_column'] = np.nan * len(real_data)
        synthetic_data['nan_column'] = np.nan * len(synthetic_data)
        metadata['columns']['nan_column'] = {'sdtype': 'numerical'}
        column_names.append('nan_column')

        report = QualityReport()

        # Run
        report.generate(real_data[column_names], synthetic_data[column_names], metadata)

        # Assert
        expected_details_column_shapes_dict = {
            'Column': [
                'start_date', 'second_perc', 'work_experience', 'degree_type', 'nan_column'
            ],
            'Metric': [
                'KSComplement', 'KSComplement', 'TVComplement', 'TVComplement', 'KSComplement'
            ],
            'Score': [
                0.7011066184294531, 0.627906976744186, 0.9720930232558139, 0.9255813953488372,
                np.nan
            ],
            'Error': [
                None, None, None, None,
                'Error: ValueError Data passed to ks_2samp must not be empty'
            ]
        }

        expected_details_cpt__dict = {
            'Column 1': [
                'start_date', 'start_date', 'start_date', 'start_date', 'second_perc',
                'second_perc', 'second_perc', 'work_experience', 'work_experience',
                'degree_type'
            ],
            'Column 2': [
                'second_perc', 'work_experience', 'degree_type', 'nan_column',
                'work_experience', 'degree_type', 'nan_column', 'degree_type',
                'nan_column', 'nan_column'
            ],
            'Metric': [
                'CorrelationSimilarity', 'ContingencySimilarity', 'ContingencySimilarity',
                'CorrelationSimilarity', 'ContingencySimilarity', 'ContingencySimilarity',
                'CorrelationSimilarity', 'ContingencySimilarity', 'ContingencySimilarity',
                'ContingencySimilarity'
            ],
            'Score': [
                0.9854510263003199, 0.586046511627907, 0.6232558139534884, np.nan,
                0.7348837209302326, 0.6976744186046512, np.nan, 0.8976744186046511,
                0.9720930232558139, 0.9255813953488372
            ],
            'Real Correlation': [
                0.04735340044317632, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan,
                np.nan, np.nan
            ],
            'Synthetic Correlation': [
                0.07645134784253645, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan,
                np.nan, np.nan
            ],
            'Error': [
                None, None, None, 'Error: ValueError x and y must have length at least 2.',
                None, None, 'Error: ValueError x and y must have length at least 2.', None,
                None, None
            ]
        }
        expected_details_column_shapes = pd.DataFrame(expected_details_column_shapes_dict)
        expected_details_cpt = pd.DataFrame(expected_details_cpt__dict)

        pd.testing.assert_frame_equal(
            report.get_details('Column Shapes'), expected_details_column_shapes
        )
        pd.testing.assert_frame_equal(
            report.get_details('Column Pair Trends'), expected_details_cpt
        )
