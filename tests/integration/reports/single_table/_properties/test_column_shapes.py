import pandas as pd
from packaging import version

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
                'start_date',
                'end_date',
                'salary',
                'duration',
                'high_perc',
                'high_spec',
                'mba_spec',
                'second_perc',
                'gender',
                'degree_perc',
                'placed',
                'experience_years',
                'employability_perc',
                'mba_perc',
                'work_experience',
                'degree_type',
            ],
            'Metric': [
                'KSComplement',
                'KSComplement',
                'KSComplement',
                'KSComplement',
                'KSComplement',
                'TVComplement',
                'TVComplement',
                'KSComplement',
                'TVComplement',
                'KSComplement',
                'TVComplement',
                'KSComplement',
                'KSComplement',
                'KSComplement',
                'TVComplement',
                'TVComplement',
            ],
            'Score': [
                0.6621621621621622,
                0.849290780141844,
                0.8531399046104928,
                0.43918918918918914,
                0.8976744186046511,
                0.9860465116279069,
                0.986046511627907,
                0.8976744186046511,
                1.0,
                0.9162790697674419,
                0.9906976744186047,
                0.3441860465116279,
                0.9348837209302325,
                0.9255813953488372,
                0.9953488372093023,
                0.9395348837209302,
            ],
        }
        expected_details = pd.DataFrame(expected_details_dict)
        pd.testing.assert_frame_equal(column_shape_property.details, expected_details)
        assert score == 0.8511084702797364

    def test_get_score_errors(self):
        """Test the ``get_score`` method when the metrics are raising errors for some columns."""
        # Setup
        real_data, synthetic_data, metadata = load_demo('single_table')

        real_data['start_date'].iloc[0] = 0
        real_data['employability_perc'].iloc[2] = 'a'

        # Run
        column_shape_property = ColumnShapes()
        score = column_shape_property.get_score(real_data, synthetic_data, metadata)

        # Assert
        pandas_version = version.parse(pd.__version__)
        if pandas_version >= version.parse('2.3.0'):
            expected_message_1 = expected_message_2 = (
                'TypeError: Expected numeric dtype, got object instead.'
            )
        elif pandas_version >= version.parse('2.2.0'):
            expected_message_1 = (
                "TypeError: '<' not supported between instances of 'Timestamp' and 'int'"
            )
            expected_message_2 = (
                "TypeError: '<' not supported between instances of 'str' and 'float'"
            )
        else:
            expected_message_1 = (
                "TypeError: unsupported operand type(s) for *: 'Timestamp' and 'float'"
            )
            expected_message_2 = "TypeError: can't multiply sequence by non-int of type 'float'"

        details = column_shape_property.details
        details_nan = details.loc[pd.isna(details['Score'])]
        column_names_nan = details_nan['Column'].tolist()
        error_messages = details_nan['Error'].tolist()
        assert column_names_nan == ['start_date', 'employability_perc']
        assert error_messages[0] == expected_message_1
        assert error_messages[1] == expected_message_2
        assert score == 0.858620688670242
