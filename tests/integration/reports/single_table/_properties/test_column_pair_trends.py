import json
import re

import numpy as np
import pandas as pd

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
        real_data = pd.read_csv('sdmetrics/demos/single_table/real.csv')[column_names]
        synthetic_data = pd.read_csv('sdmetrics/demos/single_table/synthetic.csv')[column_names]
        with open('sdmetrics/demos/single_table/metadata.json', 'r') as f:
            metadata = json.load(f)

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
                0.9854510263003199, 0.8, 0.8511627906976744, 0.627906976744186,
                0.6139534883720931, 0.8976744186046511
            ],
            'Real Correlation': [
                0.04735340044317632, np.nan, np.nan, np.nan, np.nan, np.nan
            ],
            'Synthetic Correlation': [
                0.07645134784253645, np.nan, np.nan, np.nan, np.nan, np.nan
            ]
        }
        expected_details = pd.DataFrame(expected_details_dict)
        pd.testing.assert_frame_equal(column_shape_property._details, expected_details)
        assert score == 0.796

    def test_get_score_warnings(self, recwarn):
        """Test the ``get_score`` method when the metrics are raising erros for some columns."""
        # Setup
        column_names = [
            'student_id', 'degree_type', 'start_date', 'second_perc', 'work_experience'
        ]
        real_data = pd.read_csv('sdmetrics/demos/single_table/real.csv')[column_names]
        synthetic_data = pd.read_csv('sdmetrics/demos/single_table/synthetic.csv')[column_names]
        with open('sdmetrics/demos/single_table/metadata.json', 'r') as f:
            metadata = json.load(f)

        metadata['columns'] = {
            key: val for key, val in metadata['columns'].items() if key in column_names
        }

        real_data['second_perc'].iloc[2] = 'a'

        # Run
        column_shape_property = ColumnPairTrends()

        expected_message_1 = re.escape(
            "Unable to discretize 'second_perc'. No column pair trends metric will be "
            'calculated between this column and boolean/categorical columns. Encountered '
            'Error: ValueError Unable to parse string \"a\" at position 2'
        )

        expected_message_2 = re.escape(
            "Unable to compute Column Pair Trends for column ('start_date', 'second_perc'). "
            "Encountered Error: ValueError could not convert string to float: 'a'"
        )

        score = column_shape_property.get_score(real_data, synthetic_data, metadata)

        # Assert
        assert re.match(expected_message_1, str(recwarn[0].message))
        assert re.match(expected_message_2, str(recwarn[1].message))

        details = column_shape_property._details
        column_names_nan = list(details.loc[pd.isna(details['Score'])]['Column 2'])
        assert column_names_nan == ['second_perc']
        assert score == 0.51
