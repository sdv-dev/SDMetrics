import time
from datetime import date, datetime

import numpy as np
import pandas as pd

from sdmetrics.demos import load_demo
from sdmetrics.reports.single_table import QualityReport
from tests.utils import get_error_type


def _set_thresholds_zero(report):
    report.real_correlation_threshold = 0
    report.real_association_threshold = 0


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
        _set_thresholds_zero(report)

        # Run
        report.generate(real_data, synthetic_data, metadata)
        properties = report.get_properties()

        # Assert
        pd.testing.assert_frame_equal(
            properties,
            pd.DataFrame({
                'Property': ['Column Shapes', 'Column Pair Trends'],
                'Score': [0.750000, 0.5005754481922459],
            }),
        )

    def test_report_end_to_end(self):
        """Test the quality report end to end.

        The report must compute each property and the overall quality score.
        """
        # Setup
        column_names = ['student_id', 'degree_type', 'start_date', 'second_perc', 'work_experience']
        real_data, synthetic_data, metadata = load_demo(modality='single_table')

        metadata['columns'] = {
            key: val for key, val in metadata['columns'].items() if key in column_names
        }
        report = QualityReport()
        _set_thresholds_zero(report)
        report.num_rows_subsample = None

        # Run
        generate_start_time = time.time()
        report.generate(real_data[column_names], synthetic_data[column_names], metadata)
        generate_end_time = time.time()

        # Assert
        expected_details_column_shapes_dict = {
            'Column': ['start_date', 'second_perc', 'work_experience', 'degree_type'],
            'Metric': ['KSComplement', 'KSComplement', 'TVComplement', 'TVComplement'],
            'Score': [
                0.6621621621621622,
                0.8976744186046511,
                0.9953488372093023,
                0.9395348837209302,
            ],
        }

        expected_details_cpt__dict = {
            'Column 1': [
                'start_date',
                'start_date',
                'start_date',
                'second_perc',
                'second_perc',
                'work_experience',
            ],
            'Column 2': [
                'second_perc',
                'work_experience',
                'degree_type',
                'work_experience',
                'degree_type',
                'degree_type',
            ],
            'Metric': [
                'CorrelationSimilarity',
                'ContingencySimilarity',
                'ContingencySimilarity',
                'ContingencySimilarity',
                'ContingencySimilarity',
                'ContingencySimilarity',
            ],
            'Score': [
                0.9187918131436303,
                0.6744186046511629,
                0.7162790697674419,
                0.813953488372093,
                0.772093023255814,
                0.9348837209302325,
            ],
            'Real Correlation': [0.04735340044317632, np.nan, np.nan, np.nan, np.nan, np.nan],
            'Synthetic Correlation': [
                -0.11506297326956305,
                np.nan,
                np.nan,
                np.nan,
                np.nan,
                np.nan,
            ],
            'Real Association': [np.nan] * 6,
            'Meets Threshold?': [True] * 6,
        }
        expected_details_column_shapes = pd.DataFrame(expected_details_column_shapes_dict)
        expected_details_cpt = pd.DataFrame(expected_details_cpt__dict)

        pd.testing.assert_frame_equal(
            report.get_details('Column Shapes'), expected_details_column_shapes
        )
        pd.testing.assert_frame_equal(
            report.get_details('Column Pair Trends'), expected_details_cpt
        )
        assert report.get_score() == 0.8393750143888287
        assert report._properties['Column Shapes'].num_rows_subsample is None
        assert report._properties['Column Pair Trends'].num_rows_subsample is None
        report_info = report.get_info()
        assert report_info == report.report_info

        expected_info_keys = {
            'report_type',
            'generated_date',
            'sdmetrics_version',
            'num_rows_real_data',
            'num_rows_synthetic_data',
            'generation_time',
        }
        assert report_info.keys() == expected_info_keys
        assert report_info['report_type'] == 'QualityReport'
        assert report_info['num_rows_real_data'] == 215
        assert report_info['num_rows_synthetic_data'] == 215
        assert report_info['generation_time'] <= generate_end_time - generate_start_time

    def test_column_pair_trends_threshold_changes_details(self):
        """Test threshold impact on column pair trends details."""
        # Setup
        real_data, synthetic_data, metadata = load_demo(modality='single_table')
        report_default = QualityReport()
        report_zero = QualityReport()
        _set_thresholds_zero(report_zero)

        # Run
        report_default.generate(real_data, synthetic_data, metadata, verbose=False)
        report_zero.generate(real_data, synthetic_data, metadata, verbose=False)
        score_default = (
            report_default
            .get_properties()
            .loc[lambda df: df['Property'] == 'Column Pair Trends', 'Score']
            .iloc[0]
        )
        score_zero = (
            report_zero
            .get_properties()
            .loc[lambda df: df['Property'] == 'Column Pair Trends', 'Score']
            .iloc[0]
        )

        # Assert
        assert score_zero >= score_default  # scores should be approximately 0.8 > 0.7

    def test_with_large_dataset(self):
        """Test the quality report with a large dataset (>50000 rows).

        The `real_data` and `synthetic_data` in the demo have 215 rows.
        So we augment them to be larger than 50000 rows.
        """
        # Setup
        real_data, synthetic_data, metadata = load_demo(modality='single_table')
        real_data = pd.concat([real_data] * 1000, ignore_index=True)
        synthetic_data = pd.concat([synthetic_data] * 1000, ignore_index=True)

        report_1 = QualityReport()
        report_2 = QualityReport()
        _set_thresholds_zero(report_1)
        _set_thresholds_zero(report_2)

        # Run
        report_1.generate(real_data, synthetic_data, metadata, verbose=False)
        score_1_run_1 = report_1.get_score()
        report_1.generate(real_data, synthetic_data, metadata, verbose=False)
        score_1_run_2 = report_1.get_score()
        report_2.generate(real_data, synthetic_data, metadata, verbose=False)

        # Assert
        cpt_report_1 = report_1.get_properties().iloc[1]['Score']
        cpt_report_2 = report_2.get_properties().iloc[1]['Score']
        assert report_1._properties['Column Pair Trends'].num_rows_subsample == 50000
        assert report_2._properties['Column Pair Trends'].num_rows_subsample == 50000
        assert score_1_run_1 != score_1_run_2
        assert np.isclose(score_1_run_1, score_1_run_2, atol=0.001)
        assert np.isclose(report_2.get_score(), score_1_run_1, atol=0.001)
        assert np.isclose(cpt_report_1, cpt_report_2, atol=0.001)

    def test_quality_report_with_object_datetimes(self):
        """Test the quality report with object datetimes.

        The report must compute each property and the overall quality score.
        """
        # Setup
        column_names = ['student_id', 'degree_type', 'start_date', 'second_perc', 'work_experience']
        real_data, synthetic_data, metadata = load_demo(modality='single_table')
        for column, column_meta in metadata['columns'].items():
            if column_meta['sdtype'] == 'datetime':
                dt_format = column_meta['datetime_format']
                real_data[column] = real_data[column].dt.strftime(dt_format)

        metadata['columns'] = {
            key: val for key, val in metadata['columns'].items() if key in column_names
        }
        report = QualityReport()
        _set_thresholds_zero(report)

        # Run
        report.generate(real_data[column_names], synthetic_data[column_names], metadata)

        # Assert
        expected_details_column_shapes_dict = {
            'Column': ['start_date', 'second_perc', 'work_experience', 'degree_type'],
            'Metric': ['KSComplement', 'KSComplement', 'TVComplement', 'TVComplement'],
            'Score': [
                0.6621621621621622,
                0.8976744186046511,
                0.9953488372093023,
                0.9395348837209302,
            ],
        }

        expected_details_cpt__dict = {
            'Column 1': [
                'start_date',
                'start_date',
                'start_date',
                'second_perc',
                'second_perc',
                'work_experience',
            ],
            'Column 2': [
                'second_perc',
                'work_experience',
                'degree_type',
                'work_experience',
                'degree_type',
                'degree_type',
            ],
            'Metric': [
                'CorrelationSimilarity',
                'ContingencySimilarity',
                'ContingencySimilarity',
                'ContingencySimilarity',
                'ContingencySimilarity',
                'ContingencySimilarity',
            ],
            'Score': [
                0.9187918131436303,
                0.6744186046511629,
                0.7162790697674419,
                0.813953488372093,
                0.772093023255814,
                0.9348837209302325,
            ],
            'Real Correlation': [0.04735340044317632, np.nan, np.nan, np.nan, np.nan, np.nan],
            'Synthetic Correlation': [
                -0.11506297326956305,
                np.nan,
                np.nan,
                np.nan,
                np.nan,
                np.nan,
            ],
            'Real Association': [np.nan] * 6,
            'Meets Threshold?': [True] * 6,
        }
        expected_details_column_shapes = pd.DataFrame(expected_details_column_shapes_dict)
        expected_details_cpt = pd.DataFrame(expected_details_cpt__dict)

        pd.testing.assert_frame_equal(
            report.get_details('Column Shapes'), expected_details_column_shapes
        )
        pd.testing.assert_frame_equal(
            report.get_details('Column Pair Trends'), expected_details_cpt
        )
        assert report.get_score() == 0.8393750143888287

    def test_report_end_to_end_with_errors(self):
        """Test the quality report end to end with errors in the properties computation."""
        # Setup
        column_names = ['student_id', 'degree_type', 'start_date', 'second_perc', 'work_experience']
        real_data, synthetic_data, metadata = load_demo(modality='single_table')

        metadata['columns'] = {
            key: val for key, val in metadata['columns'].items() if key in column_names
        }

        real_data['second_perc'].iloc[2] = 'a'

        report = QualityReport()
        _set_thresholds_zero(report)

        # Run
        report.generate(real_data[column_names], synthetic_data[column_names], metadata)

        # Assert
        expected_details_column_shapes_dict = {
            'Column': ['start_date', 'second_perc', 'work_experience', 'degree_type'],
            'Metric': ['KSComplement', 'KSComplement', 'TVComplement', 'TVComplement'],
            'Score': [0.6621621621621622, np.nan, 0.9953488372093023, 0.9395348837209302],
            'Error': [
                None,
                'TypeError',
                None,
                None,
            ],
        }

        expected_details_cpt__dict = {
            'Column 1': [
                'start_date',
                'start_date',
                'start_date',
                'second_perc',
                'second_perc',
                'work_experience',
            ],
            'Column 2': [
                'second_perc',
                'work_experience',
                'degree_type',
                'work_experience',
                'degree_type',
                'degree_type',
            ],
            'Metric': [
                'CorrelationSimilarity',
                'ContingencySimilarity',
                'ContingencySimilarity',
                'ContingencySimilarity',
                'ContingencySimilarity',
                'ContingencySimilarity',
            ],
            'Score': [
                np.nan,
                0.6744186046511629,
                0.7162790697674419,
                np.nan,
                np.nan,
                0.9348837209302325,
            ],
            'Real Correlation': [np.nan] * 6,
            'Synthetic Correlation': [np.nan] * 6,
            'Real Association': [np.nan] * 6,
            'Meets Threshold?': [np.nan, True, True, np.nan, np.nan, True],
            'Error': [
                'AttributeError',  # This can be either ValueError or AttributeError
                None,
                None,
                'TypeError',
                'TypeError',
                None,
            ],
        }
        expected_details_column_shapes = pd.DataFrame(expected_details_column_shapes_dict)
        expected_details_cpt = pd.DataFrame(expected_details_cpt__dict)

        # Errors may change based on versions of scipy installed
        col_shape_report = report.get_details('Column Shapes')
        col_pair_report = report.get_details('Column Pair Trends')
        col_shape_report['Error'] = col_shape_report['Error'].apply(get_error_type)
        col_pair_report['Error'] = col_pair_report['Error'].apply(get_error_type)

        pd.testing.assert_frame_equal(col_shape_report, expected_details_column_shapes)
        pd.testing.assert_frame_equal(col_pair_report[1:], expected_details_cpt[1:])
        assert report.get_score() == 0.8204378797402054

    def test_report_with_column_nan(self):
        """Test the report with column full of NaNs."""
        # Setup
        column_names = ['student_id', 'degree_type', 'start_date', 'second_perc', 'work_experience']
        real_data, synthetic_data, metadata = load_demo(modality='single_table')

        metadata['columns'] = {
            key: val for key, val in metadata['columns'].items() if key in column_names
        }

        real_data['nan_column'] = np.nan * len(real_data)
        synthetic_data['nan_column'] = np.nan * len(synthetic_data)
        metadata['columns']['nan_column'] = {'sdtype': 'numerical'}
        column_names.append('nan_column')

        report = QualityReport()
        _set_thresholds_zero(report)

        # Run
        report.generate(real_data[column_names], synthetic_data[column_names], metadata)

        # Assert
        expected_details_column_shapes_dict = {
            'Column': ['start_date', 'second_perc', 'work_experience', 'degree_type', 'nan_column'],
            'Metric': [
                'KSComplement',
                'KSComplement',
                'TVComplement',
                'TVComplement',
                'KSComplement',
            ],
            'Score': [
                0.6621621621621622,
                0.8976744186046511,
                0.9953488372093023,
                0.9395348837209302,
                np.nan,
            ],
            'Error': [
                None,
                None,
                None,
                None,
                'ValueError: Data passed to ks_2samp must not be empty',
            ],
        }

        expected_details_cpt__dict = {
            'Column 1': [
                'start_date',
                'start_date',
                'start_date',
                'start_date',
                'second_perc',
                'second_perc',
                'second_perc',
                'work_experience',
                'work_experience',
                'degree_type',
            ],
            'Column 2': [
                'second_perc',
                'work_experience',
                'degree_type',
                'nan_column',
                'work_experience',
                'degree_type',
                'nan_column',
                'degree_type',
                'nan_column',
                'nan_column',
            ],
            'Metric': [
                'CorrelationSimilarity',
                'ContingencySimilarity',
                'ContingencySimilarity',
                'CorrelationSimilarity',
                'ContingencySimilarity',
                'ContingencySimilarity',
                'CorrelationSimilarity',
                'ContingencySimilarity',
                'ContingencySimilarity',
                'ContingencySimilarity',
            ],
            'Score': [
                0.9187918131436303,
                0.6744186046511629,
                0.7162790697674419,
                np.nan,
                0.813953488372093,
                0.772093023255814,
                np.nan,
                0.9348837209302325,
                0.9953488372093023,
                0.9395348837209302,
            ],
            'Real Correlation': [
                0.04735340044317632,
                np.nan,
                np.nan,
                np.nan,
                np.nan,
                np.nan,
                np.nan,
                np.nan,
                np.nan,
                np.nan,
            ],
            'Synthetic Correlation': [
                -0.11506297326956305,
                np.nan,
                np.nan,
                np.nan,
                np.nan,
                np.nan,
                np.nan,
                np.nan,
                np.nan,
                np.nan,
            ],
            'Real Association': [np.nan] * 10,
            'Meets Threshold?': [True, True, True, np.nan, True, True, np.nan, True, True, True],
            'Error': [
                None,
                None,
                None,
                'ValueError',
                None,
                None,
                'ValueError',
                None,
                None,
                None,
            ],
        }
        expected_details_column_shapes = pd.DataFrame(expected_details_column_shapes_dict)
        expected_details_cpt = pd.DataFrame(expected_details_cpt__dict)

        col_shape_report = report.get_details('Column Shapes')
        if 'Error' not in col_shape_report:
            # Errors may not occur in certain scipy versions
            expected_details_column_shapes = expected_details_column_shapes.drop(columns=['Error'])

        # Errors may change based on versions of library installed.
        col_pair_report = report.get_details('Column Pair Trends')
        col_pair_report['Error'] = col_pair_report['Error'].apply(get_error_type)

        pd.testing.assert_frame_equal(col_shape_report, expected_details_column_shapes)
        pd.testing.assert_frame_equal(col_pair_report, expected_details_cpt)

    def test_report_with_verbose(self, capsys):
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
        _set_thresholds_zero(report)

        # Run
        report.generate(real_data, synthetic_data, metadata)
        captured = capsys.readouterr()
        output = captured.out

        # Assert
        for pattern in key_phrases:
            pattern in output

    def test_correlation_similarity_constant_real_data(self):
        """Error out when CorrelationSimilarity is used with a constant pair of columns."""
        # Setup
        data = pd.DataFrame({'col1': [1, 1, 1, 1], 'col2': [1, 1, 1, 1]})
        metadata = {'columns': {'col1': {'sdtype': 'numerical'}, 'col2': {'sdtype': 'numerical'}}}
        report = QualityReport()
        _set_thresholds_zero(report)

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
        _set_thresholds_zero(report)

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
        _set_thresholds_zero(report)

        # Run
        report.generate(data, synthetic_data, metadata)
        error_msg = report.get_details(property_name='Column Pair Trends')['Error'][0]

        # Assert
        assert error_msg == (
            "ConstantInputError: The synthetic data in columns 'col1, col2' contains "
            'a constant value. Correlation is undefined for constant data.'
        )
