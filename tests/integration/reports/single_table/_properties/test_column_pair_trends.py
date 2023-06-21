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
                0.6436432438850543, 0.8093023255813954, 0.8930232558139535, 0.6651162790697674,
                0.655813953488372, 0.8976744186046511
            ],
            'Real Correlation': [
                0.6079255273783251, np.nan, np.nan, np.nan, np.nan, np.nan
            ],
            'Synthetic Correlation': [
                -0.10478798485156637, np.nan, np.nan, np.nan, np.nan, np.nan
            ]
        }
        expected_details = pd.DataFrame(expected_details_dict)
        pd.testing.assert_frame_equal(column_shape_property._details, expected_details)
        assert score == 0.761

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

        real_data['start_date'].iloc[0] = 0
        real_data['degree_type'].iloc[2] = 'a'

        # Run
        column_shape_property = ColumnPairTrends()

        expected_message_1 = re.escape(
            "Unable to compute Column Shape for column 'start_date'. "
            "Encountered Error: TypeError '<' not supported between instances of 'str' and 'int'"
        )
        expected_message_2 = re.escape(
            "Unable to compute Column Shape for column 'degree_type'. "
            "Encountered Error: TypeError '<' not supported between instances of 'str' and 'float'"
        )

        score = column_shape_property.get_score(real_data, synthetic_data, metadata)

        # Assert
        assert re.match(expected_message_1, str(recwarn[0].message))
        assert re.match(expected_message_2, str(recwarn[1].message))

        details = column_shape_property._details
        column_names_nan = list(details.loc[pd.isna(details['Score'])]['Column name'])
        assert column_names_nan == ['start_date', 'employability_perc']
        assert score == 0.826

