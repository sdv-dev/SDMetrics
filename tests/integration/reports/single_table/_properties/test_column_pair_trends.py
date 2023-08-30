
import numpy as np
import pandas as pd

from sdmetrics.demos import load_demo
from sdmetrics.reports.single_table._properties.column_pair_trends import ColumnPairTrends


class TestColumnPairTrends:

    def test_get_score(self):
        """Test the ``get_score`` method.

        Test the method with the different possible sdtypes as well as with one primary key.
        """
        # Setup
        column_names = [
            'student_id', 'degree_type', 'start_date', 'second_perc', 'work_experience'
        ]
        real_data, synthetic_data, metadata = load_demo(modality='single_table')

        metadata['columns'] = {
            key: val for key, val in metadata['columns'].items() if key in column_names
        }

        # Run
        column_shape_property = ColumnPairTrends()
        score = column_shape_property.get_score(real_data, synthetic_data, metadata)

        # Assert
        expected_details_dict = {
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
        expected_details = pd.DataFrame(expected_details_dict)
        pd.testing.assert_frame_equal(column_shape_property.details, expected_details)
        assert score == 0.754164318336875

    def test_get_score_warnings(self, recwarn):
        """Test the ``get_score`` method when the metrics are raising erros for some columns."""
        # Setup
        column_names = [
            'student_id', 'degree_type', 'start_date', 'second_perc', 'work_experience'
        ]
        real_data, synthetic_data, metadata = load_demo(modality='single_table')

        metadata['columns'] = {
            key: val for key, val in metadata['columns'].items() if key in column_names
        }

        real_data['second_perc'].iloc[2] = 'a'

        # Run
        column_shape_property = ColumnPairTrends()

        exp_message_1 = (
            "ValueError: could not convert string to float: 'a'"
        )

        exp_message_2 = (
            "TypeError: '<=' not supported between instances of 'float' and 'str'"
        )

        exp_error_serie = pd.Series(
            [exp_message_1, None, None, exp_message_2, exp_message_2, None]
        )

        score = column_shape_property.get_score(real_data, synthetic_data, metadata)

        # Assert
        details = column_shape_property.details
        pd.testing.assert_series_equal(details['Error'], exp_error_serie, check_names=False)
        assert score == 0.7023255813953488

    def test_only_categorical_columns(self):
        """Test the ``get_score`` method when there are only categorical columns."""
        # Setup
        column_names = [
            'student_id', 'degree_type', 'gender', 'high_spec', 'work_experience'
        ]
        real_data, synthetic_data, metadata = load_demo(modality='single_table')

        metadata['columns'] = {
            key: val for key, val in metadata['columns'].items() if key in column_names
        }

        # Run
        column_shape_property = ColumnPairTrends()
        score = column_shape_property.get_score(real_data, synthetic_data, metadata)

        # Assert
        expected_details_dict = {
            'Column 1': [
                'high_spec', 'high_spec', 'high_spec', 'gender', 'gender', 'work_experience'
            ],
            'Column 2': [
                'gender', 'work_experience', 'degree_type', 'work_experience',
                'degree_type', 'degree_type'
            ],
            'Metric': ['ContingencySimilarity'] * 6,
            'Score': [
                0.8883720930232558, 0.9023255813953488, 0.7767441860465116, 0.9348837209302325,
                0.8883720930232558, 0.8976744186046511
            ],
            'Real Correlation': [np.nan] * 6,
            'Synthetic Correlation': [np.nan] * 6
        }
        expected_details = pd.DataFrame(expected_details_dict)
        pd.testing.assert_frame_equal(column_shape_property.details, expected_details)
        assert score == 0.8813953488372093
