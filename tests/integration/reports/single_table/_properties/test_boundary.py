import numpy as np
import pandas as pd

from sdmetrics.demos import load_demo
from sdmetrics.reports.single_table._properties import Boundary


class TestBoundary:

    def test_get_score(self):
        """Test the ``get_score`` method."""
        # Setup
        real_data, synthetic_data, metadata = load_demo(modality='single_table')
        boundary_property = Boundary()

        # Run
        score = boundary_property.get_score(real_data, synthetic_data, metadata)

        # Assert
        assert score == 0.9172655676537751

        expected_details = pd.DataFrame({
            'Column': [
                'start_date', 'end_date', 'salary', 'duration', 'high_perc', 'second_perc',
                'degree_perc', 'experience_years', 'employability_perc', 'mba_perc'
            ],
            'Metric': ['BoundaryAdherence'] * 10,
            'Score': [
                0.8503937007874016, 0.8615384615384616, 0.9444444444444444, 1.0,
                0.8651162790697674, 0.9255813953488372, 0.9441860465116279, 1.0,
                0.8883720930232558, 0.8930232558139535
            ]
        })

        pd.testing.assert_frame_equal(boundary_property.details, expected_details)

    def test_get_score_error(self):
        """Test the ``get_score`` method with errors."""
        # Setup
        real_data, synthetic_data, metadata = load_demo(modality='single_table')
        real_data['start_date'].iloc[0] = 0
        real_data['employability_perc'].iloc[2] = 'a'
        real_data['salary'] = np.nan

        boundary_property = Boundary()

        # Run
        score = boundary_property.get_score(real_data, synthetic_data, metadata)

        # Assert
        expected_message_1 = (
            "TypeError: '<=' not supported between instances of 'int' and 'Timestamp'"
        )
        expected_message_2 = 'InvalidDataError: All NaN values in real data.'
        expected_message_3 = (
            "TypeError: '<=' not supported between instances of 'float' and 'str'"
        )

        details = boundary_property.details
        details_nan = details.loc[pd.isna(details['Score'])]
        column_names_nan = details_nan['Column'].tolist()
        error_messages = details_nan['Error'].tolist()
        assert column_names_nan == ['start_date', 'salary', 'employability_perc']
        assert error_messages[0] == expected_message_1
        assert error_messages[1] == expected_message_2
        assert error_messages[2] == expected_message_3
        assert score == 0.9270636340403783
