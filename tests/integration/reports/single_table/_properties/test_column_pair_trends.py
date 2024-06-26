import numpy as np
import pandas as pd
from tests.utils import get_error_type

from sdmetrics.demos import load_demo
from sdmetrics.reports.single_table._properties.column_pair_trends import ColumnPairTrends


class TestColumnPairTrends:
    def test_get_score(self):
        """Test the ``get_score`` method.

        Test the method with the different possible sdtypes as well as with one primary key.
        """
        # Setup
        column_names = ['student_id', 'degree_type', 'start_date', 'second_perc', 'work_experience']
        real_data, synthetic_data, metadata = load_demo(modality='single_table')

        metadata['columns'] = {
            key: val for key, val in metadata['columns'].items() if key in column_names
        }

        # Run
        column_pair_trends = ColumnPairTrends()
        score = column_pair_trends.get_score(real_data, synthetic_data, metadata)

        # Assert
        expected_details_dict = {
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
            'Synthetic Correlation': [-0.11506297326956302, np.nan, np.nan, np.nan, np.nan, np.nan],
        }
        expected_details = pd.DataFrame(expected_details_dict)
        pd.testing.assert_frame_equal(column_pair_trends.details, expected_details)
        assert score == 0.8050699533533958

    def test_get_score_warnings(self, recwarn):
        """Test the ``get_score`` method when the metrics are raising erros for some columns."""
        # Setup
        column_names = ['student_id', 'degree_type', 'start_date', 'second_perc', 'work_experience']
        real_data, synthetic_data, metadata = load_demo(modality='single_table')

        metadata['columns'] = {
            key: val for key, val in metadata['columns'].items() if key in column_names
        }

        real_data['second_perc'].iloc[2] = 'a'

        # Run
        column_pair_trends = ColumnPairTrends()

        exp_message_1 = 'ValueError'

        exp_message_2 = 'TypeError'

        exp_error_series = pd.Series([
            exp_message_1,
            None,
            None,
            exp_message_2,
            exp_message_2,
            None,
        ])

        score = column_pair_trends.get_score(real_data, synthetic_data, metadata)

        # Assert
        details = column_pair_trends.details
        details['Error'] = details['Error'].apply(get_error_type)
        pd.testing.assert_series_equal(details['Error'], exp_error_series, check_names=False)
        assert score == 0.7751937984496124

    def test_only_categorical_columns(self):
        """Test the ``get_score`` method when there are only categorical columns."""
        # Setup
        column_names = ['student_id', 'degree_type', 'gender', 'high_spec', 'work_experience']
        real_data, synthetic_data, metadata = load_demo(modality='single_table')

        metadata['columns'] = {
            key: val for key, val in metadata['columns'].items() if key in column_names
        }

        # Run
        column_pair_trends = ColumnPairTrends()
        score = column_pair_trends.get_score(real_data, synthetic_data, metadata)

        # Assert
        expected_details_dict = {
            'Column 1': [
                'high_spec',
                'high_spec',
                'high_spec',
                'gender',
                'gender',
                'work_experience',
            ],
            'Column 2': [
                'gender',
                'work_experience',
                'degree_type',
                'work_experience',
                'degree_type',
                'degree_type',
            ],
            'Metric': ['ContingencySimilarity'] * 6,
            'Score': [
                0.9209302325581395,
                0.9627906976744186,
                0.6837209302325581,
                0.9302325581395349,
                0.9255813953488372,
                0.9348837209302325,
            ],
            'Real Correlation': [np.nan] * 6,
            'Synthetic Correlation': [np.nan] * 6,
        }
        expected_details = pd.DataFrame(expected_details_dict)
        pd.testing.assert_frame_equal(column_pair_trends.details, expected_details)
        assert score == 0.8930232558139535

    def test_with_different_indexes(self):
        """Test the property when the real and synthetic data only differ by their indexes."""
        # Setup
        real_data = pd.DataFrame({'A': [1, 2, 3], 'B': ['a', 'b', 'a']}, index=[0, 1, 2])

        synthetic_data = pd.DataFrame({'A': [1, 2, 3], 'B': ['a', 'b', 'a']}, index=[0, 4, 2])

        metadata = {'columns': {'A': {'sdtype': 'numerical'}, 'B': {'sdtype': 'categorical'}}}

        # Run
        column_pair_trends = ColumnPairTrends()
        score = column_pair_trends.get_score(real_data, synthetic_data, metadata)

        # Assert
        assert score == 1.0
