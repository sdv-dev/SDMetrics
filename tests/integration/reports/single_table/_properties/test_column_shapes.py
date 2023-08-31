import pandas as pd

from sdmetrics.demos import load_demo
from sdmetrics.reports.single_table._properties import ColumnShapes


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
        pd.testing.assert_frame_equal(column_shape_property.details, expected_details)
        assert score == 0.8155594899871002

    def test_get_score_errors(self):
        """Test the ``get_score`` method when the metrics are raising errors for some columns."""
        # Setup
        real_data, synthetic_data, metadata = load_demo('single_table')

        real_data['start_date'].iloc[0] = 0
        real_data['employability_perc'].iloc[2] = 'a'

        # Run
        column_shape_property = ColumnShapes()

        expected_message_1 = (
            "TypeError: '<' not supported between instances of 'Timestamp' and 'int'"
        )
        expected_message_2 = (
            "TypeError: '<' not supported between instances of 'str' and 'float'"
        )

        score = column_shape_property.get_score(real_data, synthetic_data, metadata)

        # Assert

        details = column_shape_property.details
        details_nan = details.loc[pd.isna(details['Score'])]
        column_names_nan = details_nan['Column'].tolist()
        error_messages = details_nan['Error'].tolist()
        assert column_names_nan == ['start_date', 'employability_perc']
        assert error_messages[0] == expected_message_1
        assert error_messages[1] == expected_message_2
        assert score == 0.8261749908947813
