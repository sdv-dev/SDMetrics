import pandas as pd

from sdmetrics.demos import load_demo
from sdmetrics.reports.single_table._properties import Coverage


class TestCoverage:

    def test_get_score(self):
        """Test the ``get_score`` method."""
        # Setup
        real_data, synthetic_data, metadata = load_demo(modality='single_table')
        coverage_property = Coverage()

        # Run
        score = coverage_property.get_score(real_data, synthetic_data, metadata)

        # Assert
        assert score == 0.9419212095491987

        expected_details = pd.DataFrame({
            'Column': [
                'start_date', 'end_date', 'salary', 'duration', 'high_perc', 'high_spec',
                'mba_spec', 'second_perc', 'gender', 'degree_perc', 'placed', 'experience_years',
                'employability_perc', 'mba_perc', 'work_experience', 'degree_type'
            ],
            'Metric': [
                'RangeCoverage', 'RangeCoverage', 'RangeCoverage', 'RangeCoverage',
                'RangeCoverage', 'CategoryCoverage', 'CategoryCoverage', 'RangeCoverage',
                'CategoryCoverage', 'RangeCoverage', 'CategoryCoverage', 'RangeCoverage',
                'RangeCoverage', 'RangeCoverage', 'CategoryCoverage', 'CategoryCoverage'
            ],
            'Score': [
                1.0, 1.0, 0.42333783783783785, 1.0, 0.9807348482826732, 1.0, 1.0, 1.0, 1.0,
                1.0, 1.0, 0.6666666666666667, 1.0, 1.0, 1.0, 1.0
            ]
        })

        pd.testing.assert_frame_equal(coverage_property.details, expected_details)

    def test_get_score_error(self):
        """Test the ``get_score`` method with errors."""
        # Setup
        real_data, synthetic_data, metadata = load_demo(modality='single_table')
        real_data['start_date'].iloc[0] = 0
        real_data['employability_perc'].iloc[2] = 'a'

        coverage_property = Coverage()

        # Run
        score = coverage_property.get_score(real_data, synthetic_data, metadata)

        # Assert
        expected_message_1 = (
            "TypeError: '<=' not supported between instances of 'int' and 'Timestamp'"
        )
        expected_message_2 = (
            "TypeError: '<=' not supported between instances of 'float' and 'str'"
        )
        details = coverage_property.details
        details_nan = details.loc[pd.isna(details['Score'])]
        column_names_nan = details_nan['Column'].tolist()
        error_messages = details_nan['Error'].tolist()
        assert column_names_nan == ['start_date', 'employability_perc']
        assert error_messages[0] == expected_message_1
        assert error_messages[1] == expected_message_2
        assert score == 0.9336242394847984
