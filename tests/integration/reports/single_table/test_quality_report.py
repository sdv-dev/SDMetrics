import contextlib
import io
import re
import time
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
            'Score': [0.750000, 0.5005754481922459],
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
        generate_start_time = time.time()
        report.generate(real_data[column_names], synthetic_data[column_names], metadata)
        generate_end_time = time.time()

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
        assert report.get_score() == 0.7804181608907237

        report_info = report.get_info()
        assert report_info == report.report_info

        expected_info_keys = {
            'report_type', 'generated_date', 'sdmetrics_version', 'num_rows_real_data',
            'num_rows_synthetic_data', 'generation_time'
        }
        assert report_info.keys() == expected_info_keys
        assert report_info['report_type'] == 'QualityReport'
        assert report_info['num_rows_real_data'] == 215
        assert report_info['num_rows_synthetic_data'] == 215
        assert report_info['generation_time'] <= generate_end_time - generate_start_time

    def test_quality_report_with_object_datetimes(self):
        """Test the quality report with object datetimes.

        The report must compute each property and the overall quality score.
        """
        # Setup
        column_names = [
            'student_id', 'degree_type', 'start_date', 'second_perc', 'work_experience'
        ]
        real_data, synthetic_data, metadata = load_demo(modality='single_table')
        for column, column_meta in metadata['columns'].items():
            if column_meta['sdtype'] == 'datetime':
                dt_format = column_meta['datetime_format']
                real_data[column] = real_data[column].dt.strftime(dt_format)

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
        assert report.get_score() == 0.7804181608907237

    def test_report_end_to_end_with_errors(self):
        """Test the quality report end to end with errors in the properties computation."""
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
                "TypeError: '<' not supported between instances of 'str' and 'float'",
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
                "ValueError: could not convert string to float: 'a'",
                None,
                None,
                "TypeError: '<=' not supported between instances of 'float' and 'str'",
                "TypeError: '<=' not supported between instances of 'float' and 'str'",
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
        assert report.get_score() == 0.7842929635366918

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
                'ValueError: Data passed to ks_2samp must not be empty'
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
                None, None, None, 'ValueError: x and y must have length at least 2.',
                None, None, 'ValueError: x and y must have length at least 2.', None,
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

    def test_report_with_verbose(self):
        """Test the report with verbose.

        Check that the report prints the correct information.
        """
        # Setup
        key_phrases = [
            r'Generating\sreport\s\.\.\.',
            r'\(1/2\)\sEvaluating\sColumn\sShapes',
            r'\(2/2\)\sEvaluating\sColumn\sPair\sTrends',
            r'Overall\sQuality\sScore:\s80\.51%',
            r'Properties:',
            r'-\sColumn\sShapes:\s81\.56%',
            r'-\sColumn\sPair\sTrends:\s79\.46%',
        ]

        real_data, synthetic_data, metadata = load_demo(modality='single_table')

        real_data['nan_column'] = np.nan * len(real_data)
        synthetic_data['nan_column'] = np.nan * len(synthetic_data)
        metadata['columns']['nan_column'] = {'sdtype': 'numerical'}

        report = QualityReport()

        # Run
        with contextlib.redirect_stdout(io.StringIO()) as my_stdout:
            report.generate(real_data, synthetic_data, metadata)

        # Assert
        for pattern in key_phrases:
            match = re.search(pattern, my_stdout.getvalue())
            assert match is not None

    def test_correlation_similarity_constant_real_data(self):
        """Error out when CorrelationSimilarity is used with a constant pair of columns."""
        # Setup
        data = pd.DataFrame({'col1': [1, 1, 1, 1], 'col2': [1, 1, 1, 1]})
        metadata = {'columns': {'col1': {'sdtype': 'numerical'}, 'col2': {'sdtype': 'numerical'}}}
        report = QualityReport()

        # Run
        report.generate(data, data, metadata)
        error_msg = report.get_details(property_name='Column Pair Trends')['Error'][0]

        # Assert
        assert error_msg == (
            "ConstantInputError: The real data in columns 'col1, col2' contains "
            'a constant value. Correlation is undefined for constant data.'
        )

    def test_correlation_similarity_one_constant_real_data_column(self):
        """Error out when CorrelationSimilarity is used with one constant column."""
        # Setup
        data = pd.DataFrame({'col1': [1, 1, 1, 1], 'col2': [1.2, 1, 1, 1]})
        metadata = {'columns': {'col1': {'sdtype': 'numerical'}, 'col2': {'sdtype': 'numerical'}}}
        report = QualityReport()

        # Run
        report.generate(data, data, metadata)
        error_msg = report.get_details(property_name='Column Pair Trends')['Error'][0]

        # Assert
        assert error_msg == (
            "ConstantInputError: The real data in column 'col1' contains "
            'a constant value. Correlation is undefined for constant data.'
        )

    def test_correlation_similarity_constant_synthetic_data(self):
        """Error out when CorrelationSimilarity is used with constant synthetic data."""
        # Setup
        data = pd.DataFrame({'col1': [2, 1, 1, 1], 'col2': [3, 1, 1, 1]})
        synthetic_data = pd.DataFrame({'col1': [1, 1, 1, 1], 'col2': [1, 1, 1, 1]})
        metadata = {'columns': {'col1': {'sdtype': 'numerical'}, 'col2': {'sdtype': 'numerical'}}}
        report = QualityReport()

        # Run
        report.generate(data, synthetic_data, metadata)
        error_msg = report.get_details(property_name='Column Pair Trends')['Error'][0]

        # Assert
        assert error_msg == (
            "ConstantInputError: The synthetic data in columns 'col1, col2' contains "
            'a constant value. Correlation is undefined for constant data.'
        )
