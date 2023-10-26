import pandas as pd

from sdmetrics.demos import load_demo
from sdmetrics.reports.single_table._properties import DataValidity


class TestDataValidity:

    def test_get_score(self):
        """Test the ``get_score`` method"""
        # Setup
        real_data, synthetic_data, metadata = load_demo('single_table')

        # Run
        data_validity_property = DataValidity()
        score = data_validity_property.get_score(real_data, synthetic_data, metadata)

        # Assert
        expected_details_dict = {
            'Column': [
                'start_date', 'end_date', 'salary', 'duration', 'student_id',
                'high_perc', 'high_spec', 'mba_spec', 'second_perc', 'gender',
                'degree_perc', 'placed', 'experience_years', 'employability_perc',
                'mba_perc', 'work_experience', 'degree_type'
            ],
            'Metric': [
                'BoundaryAdherence', 'BoundaryAdherence', 'BoundaryAdherence', 'BoundaryAdherence',
                'KeyUniqueness', 'BoundaryAdherence', 'CategoryAdherence', 'CategoryAdherence',
                'BoundaryAdherence', 'CategoryAdherence', 'BoundaryAdherence', 'CategoryAdherence',
                'BoundaryAdherence', 'BoundaryAdherence', 'BoundaryAdherence', 'CategoryAdherence',
                'CategoryAdherence'
            ],
            'Score': [
                0.8503937007874016, 0.8615384615384616, 0.9444444444444444,
                1.0, 1.0, 0.8651162790697674, 1.0, 1.0, 0.9255813953488372,
                1.0, 0.9441860465116279, 1.0, 1.0, 0.8883720930232558,
                0.8930232558139535, 1.0, 1.0
            ]
        }
        expected_details = pd.DataFrame(expected_details_dict)
        pd.testing.assert_frame_equal(data_validity_property.details, expected_details)
        assert score == 0.9513326868551618

    def test_get_score_errors(self):
        """Test the ``get_score`` method when the metrics are raising errors for some columns."""
        # Setup
        real_data, synthetic_data, metadata = load_demo('single_table')

        real_data['start_date'].iloc[0] = 0
        real_data['employability_perc'].iloc[2] = 'a'

        # Run
        data_validity_property = DataValidity()

        expected_message_1 = (
            "TypeError: '<=' not supported between instances of 'int' and 'Timestamp'"
        )
        expected_message_2 = (
            "TypeError: '<=' not supported between instances of 'float' and 'str'"
        )

        score = data_validity_property.get_score(real_data, synthetic_data, metadata)

        # Assert

        details = data_validity_property.details
        details_nan = details.loc[pd.isna(details['Score'])]
        column_names_nan = details_nan['Column'].tolist()
        error_messages = details_nan['Error'].tolist()
        assert column_names_nan == ['start_date', 'employability_perc']
        assert error_messages[0] == expected_message_1
        assert error_messages[1] == expected_message_2
        assert score == 0.9622593255151395
