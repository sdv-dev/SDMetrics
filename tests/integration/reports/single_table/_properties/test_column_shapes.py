import re

import pandas as pd

from sdmetrics.demos import load_demo
from sdmetrics.reports.single_table._properties.column_shapes import ColumnShapes


class TestColumnShapes:

    def test_get_score(self):
        # Setup
        real_data, synthetic_data, metadata = load_demo('single_table')

        # Run
        column_shape_property = ColumnShapes()
        score = column_shape_property.get_score(real_data, synthetic_data, metadata)

        # Assert
        expected_details_dict = {
            'Column': [
                'start_date', 'end_date', 'salary', 'duration', 'high_perc', 'high_spec',
                'mba_spec', 'second_perc', 'gender', 'degree_perc', 'placed', 'experience_years',
                'employability_perc', 'mba_perc', 'work_experience', 'degree_type'
            ],
            'Metric': [
                'KSComplement', 'KSComplement', 'KSComplement', 'KSComplement', 'KSComplement',
                'TVComplement', 'TVComplement', 'KSComplement', 'TVComplement', 'KSComplement',
                'TVComplement', 'KSComplement', 'KSComplement', 'KSComplement', 'TVComplement',
                'TVComplement'
            ],
            'Score': [
                0.701107, 0.768919, 0.869155, 0.826051, 0.553488, 0.902326, 0.995349, 0.627907,
                0.939535, 0.627907, 0.916279, 0.800000, 0.781395, 0.841860, 0.972093, 0.925581
            ]
        }
        expected_details = pd.DataFrame(expected_details_dict)
        pd.testing.assert_frame_equal(column_shape_property._details, expected_details)
        assert score == 0.816

    def test_get_score_warnings(self, recwarn):
        """Test the ``get_score`` method when the metrics are raising erros for some columns."""
        # Setup
        real_data, synthetic_data, metadata = load_demo('single_table')

        real_data['start_date'].iloc[0] = 0
        real_data['employability_perc'].iloc[2] = 'a'

        # Run
        column_shape_property = ColumnShapes()

        expected_message_1 = re.escape(
            "Unable to compute Column Shape for column 'start_date'. Encountered Error:"
            " TypeError '<' not supported between instances of 'Timestamp' and 'int'"
        )
        expected_message_2 = re.escape(
            "Unable to compute Column Shape for column 'employability_perc'. "
            "Encountered Error: TypeError '<' not supported between instances of 'str' and 'float'"
        )

        score = column_shape_property.get_score(real_data, synthetic_data, metadata)

        # Assert
        assert re.match(expected_message_1, str(recwarn[0].message))
        assert re.match(expected_message_2, str(recwarn[1].message))

        details = column_shape_property._details
        column_names_nan = list(details.loc[pd.isna(details['Score'])]['Column'])
        assert column_names_nan == ['start_date', 'employability_perc']
        assert score == 0.826
